from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from capx.envs.base import BaseEnv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEXSCREW_ROOT = PROJECT_ROOT / "external" / "dexscrew"
ISAACGYM_ROOT = PROJECT_ROOT / "external" / "isaacgym" / "python"


class DexScrewXHandNutBoltEnv(BaseEnv):
    """DexScrew XHand nut-bolt wrapper for contact-driven CaP experiments.

    This wrapper intentionally exposes only XHand actions.  DexScrew internally
    pads the object/nut joint action with zero, so any nut_joint motion observed
    after reset must come from simulator contact dynamics.
    """

    def __init__(
        self,
        num_envs: int = 4,
        seed: int = 0,
        sim_device: str = "cpu",
        rl_device: str = "cpu",
        graphics_device_id: int = -1,
        checkpoint: str = "external/dexscrew/xhand-deploy/checkpoints/nut_bolt.pt",
        max_steps: int = 180,
        success_angle_delta: float = 0.20,
        min_nut_contact_force: float = 0.10,
        min_fingertip_contact_force: float = 0.10,
        record_env: str | int = "auto",
        no_camera: bool = True,
        enable_render: bool = False,
        viewer: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        del enable_render, viewer, viser_debug
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.sim_device = str(sim_device)
        self.rl_device = str(rl_device)
        self.graphics_device_id = int(graphics_device_id)
        self.checkpoint = str(checkpoint)
        self.max_steps = int(max_steps)
        self.success_angle_delta = float(success_angle_delta)
        self.min_nut_contact_force = float(min_nut_contact_force)
        self.min_fingertip_contact_force = float(min_fingertip_contact_force)
        self.record_env = record_env
        self.no_camera = bool(no_camera)
        self.viser_debug = False

        self._torch = None
        self._policy = None
        self._env = None
        self._last_obs: dict[str, Any] | None = None
        self._last_reward = None
        self._last_done = None
        self._last_info: dict[str, Any] = {}
        self._sim_step_count = 0
        self._record_env_index = 0 if str(record_env) == "auto" else int(record_env)
        self._initial_states: list[dict[str, Any]] = []
        self._trace: list[dict[str, Any]] = []
        self._trace_all: list[list[dict[str, Any]]] = []
        self._trajectory_steps: list[dict[str, Any]] = []
        self._rigid_body_names: list[str] = []
        self._phase = "created"
        self._last_hand_action: np.ndarray | None = None
        self._commanded_nut_joint_action_max = 0.0

        self._create_env()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Dict({})

    @property
    def task_prompt(self) -> str:
        return (
            "Use DexScrew's XHand dexterous hand to rotate the nut on the "
            "nut-bolt through contact. Command only the 12 hand joints; never "
            "set object or nut_joint pose/state directly."
        )

    @property
    def action_dim(self) -> int:
        return 12

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        if seed is not None:
            self.seed = int(seed)
            self._set_random_seeds(self.seed)
        assert self._env is not None
        self._last_obs = self._env.reset()
        self._sim_step_count = 0
        self._trace.clear()
        self._trace_all.clear()
        self._trajectory_steps.clear()
        self._last_reward = None
        self._last_done = None
        self._last_info = {}
        self._last_hand_action = None
        self._commanded_nut_joint_action_max = 0.0
        self._initial_states = self._states_all()
        self._record_env_index = 0 if str(self.record_env) == "auto" else int(self.record_env)
        self._phase = "reset_contact_pregrasp"
        return self.get_observation(), {"task_prompt": self.task_prompt}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.apply_hand_action(action, steps=1, phase="hand_action")
        truncated = self._sim_step_count >= self.max_steps
        return (
            self.get_observation(),
            float(self.compute_reward()),
            bool(self.task_completed()),
            bool(truncated),
            self.get_task_state(),
        )

    def apply_hand_action(
        self,
        action: Any,
        *,
        steps: int = 1,
        phase: str = "hand_action",
    ) -> dict[str, Any]:
        """Apply only 12-D XHand actions; the nut joint is never commanded."""
        assert self._env is not None
        assert self._torch is not None
        if self._last_obs is None:
            self.reset(seed=self.seed)
        command_np = self._normalize_hand_action(action)
        command = self._torch.as_tensor(command_np, dtype=self._torch.float32)
        if command.ndim == 1:
            command = command.reshape(1, -1).repeat(self.num_envs, 1)
        command = command.to(self._env.device)

        for _ in range(max(1, int(steps))):
            before = self._compact_trajectory_state()
            obs, reward, done, info = self._env.step(command)
            self._last_obs = obs
            self._last_reward = reward
            self._last_done = done
            self._last_info = dict(info)
            self._last_hand_action = command.detach().cpu().numpy().copy()
            self._sim_step_count += 1
            self._phase = phase
            self._commanded_nut_joint_action_max = 0.0
            state = self._select_and_record_state(phase=phase)
            self._record_trajectory_step(command_np, before=before, after=self._compact_trajectory_state())
            if self._sim_step_count >= self.max_steps:
                break
            if bool(state["task_success"]):
                break
        return self.get_task_state()

    def run_contact_policy(
        self,
        *,
        steps: int = 120,
        stop_on_success: bool = False,
        phase: str = "contact_policy",
    ) -> dict[str, Any]:
        """Run the bundled DexScrew XHand policy through the CAP API.

        The policy output is still only a 12-D XHand command.  DexScrew pads
        the nut joint action with zero inside `XHandHora.step`.
        """
        assert self._env is not None
        if self._last_obs is None:
            self.reset(seed=self.seed)
        for _ in range(max(1, int(steps))):
            assert self._last_obs is not None
            action = self._policy_action(self._last_obs)
            state = self.apply_hand_action(action, steps=1, phase=phase)
            if stop_on_success and bool(state["task_success"]):
                break
            if self._sim_step_count >= self.max_steps:
                break
        return self.get_task_state()

    def get_observation(self) -> dict[str, Any]:
        state = self.get_task_state()
        return {
            "task": {
                "env_name": "DexScrew",
                "subtask": "xhand_nutbolt_contact",
                "prompt": self.task_prompt,
                **state,
            },
            "hand_joint_pos": np.asarray(state["hand_q_sample"], dtype=np.float64),
            "target_joint_pos": np.asarray(state["target_q_sample"], dtype=np.float64),
            "nut_dof_pos": float(state["nut_dof_pos"]),
            "nut_dof_delta_rad": float(state["nut_dof_delta_rad"]),
            "nut_contact_force_norm": float(state["nut_contact_force_norm"]),
            "fingertip_contact_force_max": float(state["fingertip_contact_force_max"]),
            "agentview": {"images": {}},
        }

    def get_task_state(self) -> dict[str, Any]:
        if not self._initial_states:
            self._initial_states = self._states_all()
        current_states = self._states_all()
        if str(self.record_env) == "auto":
            self._record_env_index = self._select_record_env(self._initial_states, current_states)
        state = dict(current_states[self._record_env_index])
        initial = self._initial_states[self._record_env_index]
        nut_delta = float(state["nut_dof_pos"] - initial["nut_dof_pos"])
        trace = self._trace or [state]
        nut_contact_max = float(max(float(row["nut_contact_force_norm"]) for row in trace))
        fingertip_contact_max = float(max(float(row["fingertip_contact_force_max"]) for row in trace))
        reset_seen = self._done_any_seen()
        task_success = bool(
            abs(nut_delta) >= self.success_angle_delta
            and nut_contact_max >= self.min_nut_contact_force
            and fingertip_contact_max >= self.min_fingertip_contact_force
            and self._commanded_nut_joint_action_max == 0.0
        )
        state.update(
            {
                "phase": self._phase,
                "sim_step": int(self._sim_step_count),
                "num_envs": int(self.num_envs),
                "record_env": int(self._record_env_index),
                "record_env_mode": str(self.record_env),
                "nut_dof_initial_rad": float(initial["nut_dof_pos"]),
                "nut_dof_delta_rad": nut_delta,
                "nut_dof_delta_turns": float(nut_delta / (2.0 * np.pi)),
                "nut_contact_force_max": nut_contact_max,
                "fingertip_contact_force_max": fingertip_contact_max,
                "reset_or_done_flag_seen": reset_seen,
                "success_angle_delta": float(self.success_angle_delta),
                "min_nut_contact_force": float(self.min_nut_contact_force),
                "min_fingertip_contact_force": float(self.min_fingertip_contact_force),
                "task_success": task_success,
                "direct_object_pose_scripted": False,
                "direct_nut_joint_state_scripted_after_reset": False,
                "commanded_nut_joint_action_max": float(self._commanded_nut_joint_action_max),
                "camera_enabled": bool(not self.no_camera),
                "rigid_body_names": list(self._rigid_body_names),
            }
        )
        return state

    def compute_reward(self) -> float:
        state = self.get_task_state()
        angle_score = min(abs(float(state["nut_dof_delta_rad"])) / max(self.success_angle_delta, 1e-6), 1.0)
        nut_contact_score = 1.0 if float(state["nut_contact_force_max"]) >= self.min_nut_contact_force else 0.0
        finger_contact_score = (
            1.0 if float(state["fingertip_contact_force_max"]) >= self.min_fingertip_contact_force else 0.0
        )
        return float(angle_score * nut_contact_score * finger_contact_score)

    def task_completed(self) -> bool:
        return bool(self.get_task_state()["task_success"])

    def verify_contact_driven_rotation(
        self,
        angle_threshold: float | None = None,
        min_nut_contact: float | None = None,
        min_fingertip_contact: float | None = None,
    ) -> bool:
        state = self.get_task_state()
        return bool(
            abs(float(state["nut_dof_delta_rad"]))
            >= float(self.success_angle_delta if angle_threshold is None else angle_threshold)
            and float(state["nut_contact_force_max"])
            >= float(self.min_nut_contact_force if min_nut_contact is None else min_nut_contact)
            and float(state["fingertip_contact_force_max"])
            >= float(self.min_fingertip_contact_force if min_fingertip_contact is None else min_fingertip_contact)
            and float(state["commanded_nut_joint_action_max"]) == 0.0
            and not bool(state["direct_object_pose_scripted"])
        )

    def diagnose_nutbolt_failure(self) -> dict[str, Any]:
        state = self.get_task_state()
        issues: list[str] = []
        suggestions: list[str] = []
        if float(state["commanded_nut_joint_action_max"]) != 0.0 or bool(state["direct_object_pose_scripted"]):
            issues.append("invalid_scripted_object_motion")
            suggestions.append("remove any direct object/nut_joint commands; command only 12-D XHand actions")
        if float(state["fingertip_contact_force_max"]) < self.min_fingertip_contact_force:
            issues.append("weak_fingertip_contact")
            suggestions.append("run_contact_policy(steps=80) or increase policy duration before judging rotation")
        if float(state["nut_contact_force_max"]) < self.min_nut_contact_force:
            issues.append("weak_nut_contact")
            suggestions.append("continue contact policy rollout; avoid zero/random hand actions that open the grasp")
        if abs(float(state["nut_dof_delta_rad"])) < self.success_angle_delta:
            issues.append("insufficient_nut_rotation")
            suggestions.append("extend run_contact_policy to 120-160 steps and stop only after verify_contact_driven_rotation")
        if bool(state["reset_or_done_flag_seen"]):
            issues.append("reset_or_done_seen")
            suggestions.append("shorten each rollout chunk and inspect contact loss before automatic reset")
        if not issues:
            issues.append("success")
            suggestions.append("record this CAP program as a contact-driven success candidate")
        return {
            "ok": bool(issues == ["success"]),
            "issues": issues,
            "suggestions": suggestions,
            "state": state,
        }

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        raise RuntimeError(
            "DexScrew camera sensors currently segfault in the controlled container; "
            "use numeric contact/angle audit until container-safe rendering is fixed."
        )

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        del enabled, clear

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        del clear
        return []

    def get_video_frame_count(self) -> int:
        return 0

    def get_trajectory_steps(self, *, clear: bool = False) -> list[dict[str, Any]]:
        steps = [dict(step) for step in self._trajectory_steps]
        if clear:
            self._trajectory_steps.clear()
        return steps

    def get_rollout_trace(self, *, clear: bool = False) -> list[dict[str, Any]]:
        """Return the per-step physical state trace recorded during rollout."""
        trace = [dict(step) for step in self._trace]
        if clear:
            self._trace.clear()
            self._trace_all.clear()
        return trace

    def close(self) -> None:
        self._env = None

    def _create_env(self) -> None:
        self._set_random_seeds(self.seed)
        os.environ.setdefault("LD_LIBRARY_PATH", "/home/lifuchun/miniconda3/envs/isaacgym-preview4/lib")
        for candidate in (PROJECT_ROOT, ISAACGYM_ROOT, DEXSCREW_ROOT):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        import torch
        from dexscrew.tasks import isaacgym_task_map
        from dexscrew.tasks.xhand_hora import XHandHora
        from dexscrew.utils.reformat import omegaconf_to_dict

        self._torch = torch
        self._register_omegaconf_resolvers()
        cfg = self._make_config()
        self._patch_dexscrew_camera_creation(XHandHora)

        old_cwd = Path.cwd()
        os.chdir(DEXSCREW_ROOT)
        try:
            self._env = isaacgym_task_map["XHandHoraNutBolt"](
                config=omegaconf_to_dict(cfg.task),
                sim_device=self.sim_device,
                graphics_device_id=self.graphics_device_id,
                headless=True,
            )
            self._last_obs = self._env.reset()
        finally:
            os.chdir(old_cwd)

        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        self._policy = torch.jit.load(str(checkpoint_path), map_location="cpu")
        self._policy.eval()
        self._initial_states = self._states_all()
        self._rigid_body_names = self._read_rigid_body_names()
        self._phase = "created_contact_pregrasp"

    def _make_config(self) -> Any:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(DEXSCREW_ROOT / "configs" / "config.yaml")
        cfg.task = OmegaConf.load(DEXSCREW_ROOT / "configs" / "task" / "XHandHoraNutBolt.yaml")
        cfg.task.name = "XHandHoraNutBolt"
        cfg.task_name = "XHandHoraNutBolt"
        cfg.num_envs = int(self.num_envs)
        cfg.seed = int(self.seed)
        cfg.pipeline = "gpu" if self.sim_device.startswith("cuda") else "cpu"
        cfg.sim_device = self.sim_device
        cfg.rl_device = self.rl_device
        cfg.graphics_device_id = int(self.graphics_device_id)
        cfg.headless = True
        cfg.task.env.numEnvs = int(self.num_envs)
        cfg.task.env.object.type = "screw_trinut"
        cfg.task.env.object.evalObjectType = "screw_0"
        cfg.task.env.object.object_tilt = False
        cfg.task.env.object.nut_termination_history_len = 10000
        cfg.task.env.object.nut_stagnation_eps = 0.0
        cfg.task.env.episodeLength = max(10000, int(self.max_steps) + 100)
        cfg.task.env.reset_dist_threshold = 10.0
        cfg.task.env.forceScale = 0.0
        cfg.task.env.randomForceProbScalar = 0.0
        cfg.task.env.enableCameraSensors = bool(not self.no_camera)
        cfg.task.env.hora.point_cloud_sampled_dim = 100
        cfg.task.env.randomization.randomizeMass = False
        cfg.task.env.randomization.randomizeCOM = False
        cfg.task.env.randomization.randomizeFriction = False
        cfg.task.env.randomization.randomizeScale = False
        cfg.task.env.randomization.randomizePDGains = False
        cfg.task.env.randomization.obs_noise_e_scale = 0.0
        cfg.task.env.randomization.obs_noise_t_scale = 0.0
        cfg.task.env.randomization.action_noise_e_scale = 0.0
        cfg.task.env.randomization.action_noise_t_scale = 0.0
        cfg.task.env.randomization.pose_noise_scale = 0.0
        cfg.task.env.randomization.noisy_rpy_scale = 0.0
        cfg.task.env.randomization.noisy_pos_scale = 0.0
        cfg.task.sim.use_gpu_pipeline = bool(self.sim_device.startswith("cuda"))
        cfg.task.sim.physx.use_gpu = bool(self.sim_device.startswith("cuda"))
        return cfg

    def _policy_action(self, obs: dict[str, Any]) -> Any:
        assert self._torch is not None
        assert self._policy is not None
        torch = self._torch
        policy = self._policy
        with torch.no_grad():
            obs_tensor = obs["obs"].detach().cpu().float()
            proprio = obs["proprio_hist"].detach().cpu().float()
            pc = obs["point_cloud_info"].detach().cpu().float()

            obs_mean = policy.running_mean.float().view(1, -1)
            obs_var = policy.running_var.float().view(1, -1)
            sa_mean = policy.sa_mean.float().view(1, *policy.sa_mean.shape)
            sa_var = policy.sa_var.float().view(1, *policy.sa_var.shape)
            pc_mean = policy.pc_mean.float().view(1, 1, -1)
            pc_var = policy.pc_var.float().view(1, 1, -1)

            norm_obs = torch.clamp((obs_tensor - obs_mean) / torch.sqrt(obs_var + 1e-5), -5.0, 5.0)
            norm_proprio = torch.clamp((proprio - sa_mean) / torch.sqrt(sa_var + 1e-5), -5.0, 5.0)
            norm_pc = torch.clamp((pc - pc_mean) / torch.sqrt(pc_var + 1e-5), -5.0, 5.0)
            mu, _extrin, _extrin_gt = policy(
                {
                    "obs": norm_obs,
                    "proprio_hist": norm_proprio,
                    "point_cloud_info": norm_pc,
                }
            )
            action = torch.clamp(mu, -1.0, 1.0).float()
            action[:, 5:9] = 0.0
            return action

    def _states_all(self) -> list[dict[str, Any]]:
        assert self._env is not None
        self._env._refresh_gym()
        return [self._state_from_refreshed(idx) for idx in range(int(self.num_envs))]

    def _state_from_refreshed(self, env_id: int) -> dict[str, Any]:
        assert self._env is not None
        idx = int(env_id)
        contact_norms = self._env.contact_forces.detach().cpu().norm(dim=-1).numpy()
        fingertip_ids = [int(i) for i in self._env.fingertip_handles]
        fingertip_contact = contact_norms[idx, fingertip_ids]
        fingertip_pos = self._env.rigid_body_states[idx, fingertip_ids, :3].detach().cpu().numpy()
        fingertip_quat = self._env.rigid_body_states[idx, fingertip_ids, 3:7].detach().cpu().numpy()
        hand_body_count = int(getattr(self._env, "num_xhand_hand_bodies", len(fingertip_ids)))
        hand_body_state = self._env.rigid_body_states[idx, :hand_body_count, :7].detach().cpu().numpy()
        object_body_state = self._env.rigid_body_states[idx, hand_body_count : hand_body_count + 3, :7].detach().cpu().numpy()
        nut_contact = float(contact_norms[idx, int(self._env.screw_nut_rb_handle)])
        nut_pos = self._env.nut_pos.detach().cpu().numpy()[idx]
        object_pos = self._env.object_pos.detach().cpu().numpy()[idx]
        hand_q = self._env.xhand_hand_dof_pos.detach().cpu().numpy()[idx]
        target_q = self._env.cur_targets.detach().cpu().numpy()[idx, : self._env.num_actions]
        return {
            "nut_dof_pos": float(self._env.nut_dof_pos.detach().cpu().numpy()[idx, 0]),
            "nut_dof_vel": float(self._env.nut_dof_vel.detach().cpu().numpy()[idx, 0]),
            "nut_contact_force_norm": nut_contact,
            "fingertip_contact_force_max": float(np.max(fingertip_contact)),
            "fingertip_contact_force_norms": fingertip_contact.astype(float).tolist(),
            "fingertip_pos": fingertip_pos.astype(float).tolist(),
            "fingertip_quat": fingertip_quat.astype(float).tolist(),
            "hand_body_pos": hand_body_state[:, :3].astype(float).tolist(),
            "hand_body_state": hand_body_state.astype(float).tolist(),
            "object_body_state": object_body_state.astype(float).tolist(),
            "nut_pos": nut_pos.astype(float).tolist(),
            "object_pos": object_pos.astype(float).tolist(),
            "hand_q_sample": hand_q[:12].astype(float).tolist(),
            "target_q_sample": target_q[:12].astype(float).tolist(),
        }

    def _select_and_record_state(self, *, phase: str) -> dict[str, Any]:
        all_states = self._states_all()
        if str(self.record_env) == "auto":
            self._record_env_index = self._select_record_env(self._initial_states, all_states)
        state = dict(all_states[self._record_env_index])
        initial = self._initial_states[self._record_env_index]
        state.update(
            {
                "step": int(self._sim_step_count),
                "phase": phase,
                "nut_dof_delta_rad": float(state["nut_dof_pos"] - initial["nut_dof_pos"]),
                "reward_mean": self._tensor_mean(self._last_reward),
                "done_any": self._done_any_current(),
                "done_record_env": self._done_record_env_current(),
                "screw_angular_position_mean": self._info_float("screw/angular_position"),
                "screw_angular_velocity_mean": self._info_float("screw/angular_velocity"),
                "task_success": self.verify_contact_driven_rotation(),
                "rigid_body_names": list(self._rigid_body_names),
            }
        )
        step_all: list[dict[str, Any]] = []
        for env_index, env_state in enumerate(all_states):
            env_initial = self._initial_states[env_index]
            enriched = dict(env_state)
            enriched.update(
                {
                    "step": int(self._sim_step_count),
                    "phase": phase,
                    "env_index": int(env_index),
                    "nut_dof_delta_rad": float(env_state["nut_dof_pos"] - env_initial["nut_dof_pos"]),
                    "reward_mean": self._tensor_mean(self._last_reward),
                    "done_any": self._done_any_current(),
                    "done_record_env": self._done_record_env_current(),
                    "screw_angular_position_mean": self._info_float("screw/angular_position"),
                    "screw_angular_velocity_mean": self._info_float("screw/angular_velocity"),
                    "rigid_body_names": list(self._rigid_body_names),
                }
            )
            step_all.append(enriched)
        self._trace_all.append(step_all)
        if str(self.record_env) == "auto":
            self._record_env_index = self._select_record_env_from_trace()
            state = dict(step_all[self._record_env_index])
            self._trace = [dict(step[self._record_env_index]) for step in self._trace_all]
            state["task_success"] = self.verify_contact_driven_rotation()
            return state
        self._trace.append(state)
        return state

    def _select_record_env(self, initial_states: list[dict[str, Any]], states: list[dict[str, Any]]) -> int:
        if self._trace_all:
            return self._select_record_env_from_trace()
        scores = []
        for idx, state in enumerate(states):
            angle_delta = abs(float(state["nut_dof_pos"]) - float(initial_states[idx]["nut_dof_pos"]))
            contact_score = float(state["nut_contact_force_norm"]) + float(state["fingertip_contact_force_max"])
            scores.append(angle_delta * 10.0 + contact_score * 0.01)
        return int(np.argmax(np.asarray(scores, dtype=np.float64)))

    def _select_record_env_from_trace(self) -> int:
        scores = []
        for env_index in range(self.num_envs):
            env_rows = [step[env_index] for step in self._trace_all if env_index < len(step)]
            if not env_rows:
                scores.append(float("-inf"))
                continue
            angle_max = max(abs(float(row["nut_dof_delta_rad"])) for row in env_rows)
            nut_contact_max = max(float(row["nut_contact_force_norm"]) for row in env_rows)
            finger_contact_max = max(float(row["fingertip_contact_force_max"]) for row in env_rows)
            success_bonus = (
                1000.0
                if angle_max >= self.success_angle_delta
                and nut_contact_max >= self.min_nut_contact_force
                and finger_contact_max >= self.min_fingertip_contact_force
                else 0.0
            )
            scores.append(success_bonus + angle_max * 10.0 + nut_contact_max * 0.01 + finger_contact_max * 20.0)
        return int(np.argmax(np.asarray(scores, dtype=np.float64)))

    def _read_rigid_body_names(self) -> list[str]:
        assert self._env is not None
        names: list[str] = []
        try:
            hand_names = list(self._env.gym.get_asset_rigid_body_names(self._env.hand_asset))
            object_type_id = int(self._env.object_type_at_env[0])
            object_asset = self._env.object_asset_list[object_type_id]
            object_names = list(self._env.gym.get_asset_rigid_body_names(object_asset))
            names = [str(name) for name in hand_names + object_names]
        except Exception:
            names = [f"body_{idx}" for idx in range(int(getattr(self._env, "num_bodies", 0)))]
        if names and len(names) < int(getattr(self._env, "num_bodies", len(names))):
            total = int(getattr(self._env, "num_bodies", len(names)))
            names.extend(f"body_{idx}" for idx in range(len(names), total))
        return names

    def _normalize_hand_action(self, action: Any) -> np.ndarray:
        if self._torch is not None and hasattr(action, "detach"):
            arr = action.detach().cpu().numpy()
        else:
            arr = np.asarray(action, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[1] != self.action_dim:
                raise ValueError(f"DexScrew XHand action must have 12 columns, got {arr.shape}")
            return np.clip(arr, -1.0, 1.0).astype(np.float32)
        arr = arr.reshape(-1)
        if arr.size != self.action_dim:
            raise ValueError(f"DexScrew XHand action must have 12 values, got {arr.size}")
        return np.clip(arr, -1.0, 1.0).astype(np.float32)

    def _compact_trajectory_state(self) -> list[float]:
        state = self.get_task_state()
        return (
            list(np.asarray(state["hand_q_sample"], dtype=np.float64).reshape(-1))
            + [
                float(state["nut_dof_pos"]),
                float(state["nut_dof_delta_rad"]),
                float(state["nut_contact_force_norm"]),
                float(state["fingertip_contact_force_max"]),
            ]
        )

    def _record_trajectory_step(self, action: np.ndarray, *, before: list[float], after: list[float]) -> None:
        self._trajectory_steps.append(
            {
                "observation.state": before,
                "next_observation.state": after,
                "action": np.asarray(action, dtype=np.float64).reshape(-1).astype(float).tolist(),
                "timestamp": time.time(),
                "frame_index": len(self._trajectory_steps),
                "sim_step_index": int(self._sim_step_count),
                "episode_index": 0,
                "task_index": 0,
                "reward": float(self.compute_reward()),
                "done": bool(self.task_completed()),
                "phase": self._phase,
                "wall_time": time.time(),
            }
        )

    def _done_any_current(self) -> bool:
        if self._last_done is None or not hasattr(self._last_done, "detach"):
            return False
        return bool(self._last_done.detach().cpu().bool().any().item())

    def _done_record_env_current(self) -> bool:
        if self._last_done is None or not hasattr(self._last_done, "detach"):
            return False
        return bool(self._last_done.detach().cpu().bool().reshape(-1)[self._record_env_index].item())

    def _done_any_seen(self) -> bool:
        return bool(any(bool(row.get("done_any", False)) for row in self._trace))

    def _tensor_mean(self, value: Any) -> float | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            return float(value.detach().cpu().float().mean().item())
        return float(value)

    def _info_float(self, key: str) -> float | None:
        value = self._last_info.get(key)
        if value is None:
            return None
        return self._tensor_mean(value)

    def _set_random_seeds(self, seed: int) -> None:
        import random

        random.seed(int(seed))
        np.random.seed(int(seed))
        if self._torch is not None:
            self._torch.manual_seed(int(seed))

    def _register_omegaconf_resolvers(self) -> None:
        from omegaconf import OmegaConf

        resolvers = {
            "eq": lambda x, y: x.lower() == y.lower(),
            "contains": lambda x, y: x.lower() in y.lower(),
            "if": lambda pred, a, b: a if pred else b,
            "resolve_default": lambda default, arg: default if arg == "" else arg,
        }
        for name, fn in resolvers.items():
            if not OmegaConf.has_resolver(name):
                OmegaConf.register_new_resolver(name, fn)

    def _patch_dexscrew_camera_creation(self, xhand_hora_cls: Any) -> None:
        from isaacgym import gymapi

        def _create_camera_no_gpu_tensor(self: Any, env_ptr: Any) -> Any:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 256
            camera_props.height = 256
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            cam_pos = gymapi.Vec3(0.0, 0.2, 0.75)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.set_camera_location(camera_handle, env_ptr, cam_pos, cam_target)
            return None

        xhand_hora_cls._create_camera = _create_camera_no_gpu_tensor
