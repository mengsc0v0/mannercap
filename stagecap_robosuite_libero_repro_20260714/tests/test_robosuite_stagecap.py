from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from capx.envs.launch import LaunchArgs
from capx.envs.stage_trial import (
    _bound_stage_action_contracts,
    _grounding_snapshot_satisfied,
    _normalize_reserved_stage_locals,
    _preflight_stage_code,
    _query_stage_model_with_retries,
    _robosuite_stage_fast_path_code,
)
from capx.integrations.franka.robosuite_stage import (
    FrankaRobosuiteCubeLiftStageApi,
    FrankaRobosuiteHandoverStageApi,
    FrankaRobosuiteNutAssemblyStageApi,
    FrankaRobosuiteRestackStageApi,
    FrankaRobosuiteSpillWipeStageApi,
    FrankaRobosuiteStackStageApi,
    FrankaRobosuiteTwoArmLiftStageApi,
)
from capx.llm.client import ModelQueryArgs
from capx.utils.launch_utils import _load_config
from scripts.run_robosuite_stagecap_batch import (
    ROBOSUITE_TASKS,
    RoboSuiteBatchArgs,
    _build_config,
    _load_yaml,
    _model_output_dir,
    _validate_condition_config,
)
from vlm_planner_cap.pattern_priors import build_pattern_plan
from vlm_planner_cap.scene_snapshot import collect_readonly_scene_snapshot
from vlm_planner_cap.schema import (
    ROBOSUITE_PRIVILEGED_OBSERVATIONS,
    ROBOSUITE_PRIVILEGED_PRIMITIVES,
    ROBOSUITE_PRIVILEGED_SKILLS,
    TaskFamily,
    build_bound_api_policy,
)
from vlm_planner_cap.stage_loop import build_stage_code_prompt


TASK_CASES = (
    (
        "pick up the red cube and lift it",
        TaskFamily.CUBE_LIFT,
        ("cube_lift_ground_object_target", "cube_lift_pick_verify"),
    ),
    (
        "restack the cubes by moving green off red, then stacking red on green",
        TaskFamily.RESTACK,
        (
            "restack_ground_object_target",
            "restack_relocate_covering_cube",
            "restack_pick_lift_verify",
            "restack_transport_guarded",
            "restack_low_release_verify",
        ),
    ),
    (
        "pick up the red cube and gently stack it on top of the green cube",
        TaskFamily.STACK,
        ("stack_ground_object_target", "stack_pick_lift_verify", "stack_transport_guarded", "stack_low_release_verify"),
    ),
    (
        "grasp and insert the brown square nut onto the brown square block",
        TaskFamily.PEG_INSERTION,
        ("peg_ground_object_target", "peg_pick_lift_verify", "peg_align_guarded", "peg_insert_verify"),
    ),
    (
        "coordinate two arms to lift a pot by grasping each handle and lifting simultaneously",
        TaskFamily.BIMANUAL_LIFT,
        ("bimanual_ground_object_target", "bimanual_pregrasp_handles", "bimanual_grasp_verify", "bimanual_lift_verify"),
    ),
    (
        "wipe up the brown spill with the attached wiping tool",
        TaskFamily.WIPE,
        ("wipe_ground_object_target", "wipe_approach_surface", "wipe_execute_verify"),
    ),
    (
        "arm 0 should lift the hammer and hand it over to arm 1 by the handle",
        TaskFamily.BIMANUAL_HANDOVER,
        (
            "handover_ground_object_target",
            "handover_arm0_pick_verify",
            "handover_present_guarded",
            "handover_arm1_capture_verify",
            "handover_release_verify",
        ),
    ),
)


@pytest.mark.parametrize(("task", "family", "stage_ids"), TASK_CASES)
def test_robosuite_tasks_have_expected_stage_machines(task, family, stage_ids):
    plan = build_pattern_plan(task)

    assert family in plan.families
    assert tuple(stage.stage_id for stage in plan.stages) == stage_ids
    assert all(stage.family == family for stage in plan.stages)


def _full_robosuite_policy():
    names = set(ROBOSUITE_PRIVILEGED_PRIMITIVES)
    names.update(ROBOSUITE_PRIVILEGED_OBSERVATIONS)
    names.update(ROBOSUITE_PRIVILEGED_SKILLS)
    names.add("get_robosuite_task_context")
    return build_bound_api_policy(
        {name: object() for name in names}, privileged=True, sam3_enabled=False
    )


def test_bound_policy_is_privileged_readonly_without_sam3():
    policy = _full_robosuite_policy()
    allowed = policy.allowed_primitives + policy.allowed_observations + policy.allowed_skills

    assert policy.profile == "robosuite_privileged_readonly"
    assert not any("sam3" in name.lower() for name in allowed)
    assert "sim.data.qpos" in policy.forbidden_symbols
    assert "set_state" in policy.forbidden_symbols


def test_all_robosuite_action_fast_paths_pass_clean_code_preflight():
    policy = _full_robosuite_policy()
    checked = 0

    for task, _family, _stage_ids in TASK_CASES:
        for stage in build_pattern_plan(task).stages:
            code = _robosuite_stage_fast_path_code(stage)
            if not code:
                assert stage.stage_id.endswith("ground_object_target")
                continue
            assert not _preflight_stage_code(
                code,
                policy=policy,
                stage_id=stage.stage_id,
                allow_direct_pick_apis=True,
                stage_allowed_actions=tuple(stage.allowed_primitives),
            )
            checked += 1

    assert checked == 20


def test_robosuite_stage_contract_rejects_manual_action_substitution():
    policy = _full_robosuite_policy()
    task = next(task for task, family, _ids in TASK_CASES if family == TaskFamily.PEG_INSERTION)
    stage = next(
        stage for stage in build_pattern_plan(task).stages if stage.stage_id == "peg_pick_lift_verify"
    )
    code = '''import json
goto_pose([0.4, 0.1, 0.2], [1.0, 0.0, 0.0, 0.0])
status = "satisfied"
evidence = "manual motion completed"
repair = ""
print("STAGE_STATUS " + json.dumps({"stage_id": STAGE_ID, "status": status, "evidence": evidence, "repair": repair}, default=str))
'''

    error = _preflight_stage_code(
        code,
        policy=policy,
        stage_id=stage.stage_id,
        stage_allowed_actions=tuple(stage.allowed_primitives),
    )

    assert "StageCaP action contract violation" in error
    assert "verified_grasp_nut_handle" in error
    assert "goto_pose" in error


def test_robosuite_prompt_includes_bound_current_stage_signature():
    task = next(task for task, family, _ids in TASK_CASES if family == TaskFamily.PEG_INSERTION)
    plan = build_pattern_plan(task)
    stage = next(stage for stage in plan.stages if stage.stage_id == "peg_pick_lift_verify")

    def verified_grasp_nut_handle(*, stage_attempt: int = 1):
        """Grasp the square nut through the audited handle contract."""

    contracts = _bound_stage_action_contracts(
        {"verified_grasp_nut_handle": verified_grasp_nut_handle},
        stage,
    )
    prompt = build_stage_code_prompt(
        task,
        plan,
        stage,
        policy=_full_robosuite_policy(),
        required_action_contracts=contracts,
    )

    assert "verified_grasp_nut_handle(" in prompt
    assert "stage_attempt" in prompt
    assert "= 1" in prompt
    assert "all other robot actions are forbidden" in prompt
    assert "do not replace it with lower-level motion primitives" in prompt


def test_stage_model_transport_retries_empty_content(monkeypatch):
    calls = 0
    valid_code = '''import json
status = "satisfied"
evidence = "ok"
repair = ""
print("STAGE_STATUS " + json.dumps({"stage_id": STAGE_ID, "status": status, "evidence": evidence, "repair": repair}, default=str))
'''

    def fake_query_model(_args, _prompt):
        nonlocal calls
        calls += 1
        return {"content": "" if calls == 1 else valid_code, "reasoning": ""}

    monkeypatch.setattr("capx.envs.stage_trial.query_model", fake_query_model)
    response, raw_code, error, attempts = _query_stage_model_with_retries(
        ModelQueryArgs(model="test", server_url="http://127.0.0.1:1"),
        [{"role": "user", "content": "generate code"}],
        max_query_attempts=2,
    )

    assert calls == 2
    assert attempts == 2
    assert not error
    assert raw_code == valid_code
    assert response["content"] == valid_code


def test_stage_local_normalizer_preserves_task_status_key():
    code = '''import json
result = wipe_spill_and_verify(stage_attempt=STAGE_ATTEMPT)
task_status = get_task_status()
task_completed = bool(task_status.get("task_completed", False))
status = "satisfied" if task_completed else "needs_repair"
evidence = str({"task_completed": task_status.get("task_completed", False)})
repair = "" if task_completed else str(result.get("reason", "retry"))
print("STAGE_STATUS " + json.dumps({"stage_id": STAGE_ID, "status": status, "evidence": evidence, "repair": repair}, default=str))
'''

    normalized = _normalize_reserved_stage_locals(code)

    assert "completed = bool" in normalized
    assert "if completed" in normalized
    assert "task_completed =" not in normalized
    assert "'task_completed'" in normalized
    assert not _preflight_stage_code(
        normalized,
        policy=_full_robosuite_policy(),
        stage_id="wipe_execute_verify",
        stage_allowed_actions=("wipe_spill_and_verify",),
    )


def test_cube_grasp_state_accepts_lifted_contact_supported_grasp():
    api = FrankaRobosuiteStackStageApi.__new__(FrankaRobosuiteStackStageApi)
    api._cube_pose = lambda _name: (
        np.array([0.48, 0.09, 0.069]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.05, 0.05, 0.05]),
    )
    api.get_ee_pose = lambda: (
        np.array([0.48, 0.09, 0.187]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    api.get_gripper_state = lambda: {"opening": 0.55}
    api._contact_count = lambda *_args: 6

    state = api.measure_object_grasp_state("red cube")

    assert state["held"]
    assert state["strict_held"]
    assert state["held_by_contact"]
    assert state["elevated_contact"]
    assert state["object_ee_distance"] == pytest.approx(0.118)


def test_bimanual_diagnostic_uses_tcp_compensated_contact_evidence():
    api = FrankaRobosuiteTwoArmLiftStageApi.__new__(FrankaRobosuiteTwoArmLiftStageApi)
    handle0 = np.array([0.43, -0.01, 0.008])
    handle1 = np.array([0.74, 0.01, 0.008])
    quaternion = np.array([0.7, 0.0, 0.0, 0.714])
    api.get_handle0_pose = lambda: (handle0, quaternion)
    api.get_handle1_pose = lambda: (handle1, quaternion)
    api.get_arm0_gripper_pose = lambda: (
        handle0 + np.array([0.002, 0.0, 0.108]),
        quaternion,
    )
    api.get_arm1_gripper_pose = lambda: (
        handle1 + np.array([-0.002, 0.0, 0.108]),
        quaternion,
    )
    api.get_pot_pose = lambda: (np.array([0.58, 0.0, -0.052]), quaternion)
    api.get_gripper_state = lambda: {
        "arm0": {"opening": 0.25},
        "arm1": {"opening": 0.32},
    }
    api._contact_count = lambda *_args: 5
    api.get_task_status = lambda: {"reward": 0.4, "task_completed": False}

    diagnostic = api.get_bimanual_lift_diagnostic()

    assert diagnostic["arm0_tcp_alignment_error"] < 0.005
    assert diagnostic["arm1_tcp_alignment_error"] < 0.005
    assert diagnostic["both_grasped"]


@pytest.mark.parametrize(
    ("opening", "expected"),
    ((0.39, True), (0.29, True), (0.79, False), (0.225, False)),
)
def test_nut_transport_stability_rejects_loose_grasp(opening, expected):
    api = FrankaRobosuiteNutAssemblyStageApi.__new__(FrankaRobosuiteNutAssemblyStageApi)
    api._nut_pose = lambda name: (
        np.array([0.5, 0.0, 0.08])
        if "center" in name
        else np.array([0.45, 0.0, 0.075]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    api.get_ee_pose = lambda: (
        np.array([0.45, 0.0, 0.183]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    api.get_gripper_state = lambda: {"opening": opening}
    api._contact_count = lambda *_args: 9

    state = api.measure_object_grasp_state()

    assert state["held"]
    assert state["grasp_contact_stable"] is expected
    assert state["transport_stable_held"] is expected


def test_robosuite_snapshot_skips_unbound_and_fixed_object_grasp_calls():
    calls: list[tuple[str, str]] = []

    def grasp_state(name):
        calls.append(("state", name))
        return {"held": False}

    def affordance(name):
        calls.append(("affordance", name))
        return {"candidates": []}

    api_functions = {
        "get_task_status": lambda: {"reward": 0.0, "task_completed": False},
        "get_robosuite_task_context": lambda: {"task_family": "peg_insertion"},
        "get_instance_object_names": lambda: [
            "brown square nut",
            "brown square nut handle",
            "brown square block",
        ],
        "get_all_object_poses": lambda: {
            "brown square nut": {},
            "brown square nut handle": {},
            "brown square block": {},
        },
        "get_object_pose": lambda _name: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        "measure_object_grasp_state": grasp_state,
        "get_object_grasp_affordance": affordance,
        "get_contact_diagnostics": lambda: [],
    }
    task = next(task for task, family, _ids in TASK_CASES if family == TaskFamily.PEG_INSERTION)
    stage = build_pattern_plan(task).stages[0]

    snapshot = collect_readonly_scene_snapshot(api_functions, stage=stage, task=task)

    assert "errors" not in snapshot
    assert "grasp_state" not in snapshot["objects_of_interest"]["brown square block"]
    assert "grasp_affordance" not in snapshot["objects_of_interest"]["brown square block"]
    assert not any(name == "brown square block" for _kind, name in calls)


def test_bimanual_grounding_prefers_exact_handle_over_generic_pot_match():
    task = next(task for task, family, _ids in TASK_CASES if family == TaskFamily.BIMANUAL_LIFT)
    stage = build_pattern_plan(task).stages[0]
    snapshot = {
        "objects_of_interest": {
            "pot": {"pose": [0.0, 0.0, 0.0]},
            "pot handle 0": {"grasp_affordance": {"candidate_count": 1}},
            "pot handle 1": {"grasp_affordance": {"candidate_count": 1}},
        }
    }

    assert _grounding_snapshot_satisfied(stage, snapshot)


def test_wipe_serpentine_path_covers_both_bounds():
    points = np.array([[0.40, -0.08, 0.0], [0.62, 0.09, 0.0]])

    path = FrankaRobosuiteSpillWipeStageApi._serpentine_wipe_path(points, 0.02)
    xy = np.stack([point[:2] for point in path])

    assert len(path) >= 4
    assert xy[:, 0].min() <= 0.385
    assert xy[:, 0].max() >= 0.635
    assert xy[:, 1].min() <= -0.095
    assert xy[:, 1].max() >= 0.095


def test_wipe_marker_clustering_covers_every_input_point():
    points = np.array(
        [
            [0.40, -0.08, -0.022],
            [0.41, -0.08, -0.022],
            [0.50, 0.02, -0.022],
            [0.51, 0.02, -0.022],
        ]
    )

    targets = FrankaRobosuiteSpillWipeStageApi._cluster_marker_targets(points, 0.02)
    target_xy = np.stack([target[:2] for target in targets])

    assert len(targets) == 2
    assert all(np.min(np.linalg.norm(target_xy - point[:2], axis=1)) <= 0.02 for point in points)


@pytest.mark.parametrize(
    "config_path",
    (
        "env_configs/cube_stack/franka_robosuite_cube_stack_stagecap_privileged.yaml",
        "env_configs/cube_lifting/franka_robosuite_cube_lifting_stagecap_privileged.yaml",
        "env_configs/cube_restack/franka_robosuite_cube_restack_stagecap_privileged.yaml",
        "env_configs/nut_assembly/franka_robosuite_nut_assembly_stagecap_privileged.yaml",
        "env_configs/spill_wipe/franka_robosuite_spill_wipe_stagecap_privileged.yaml",
        "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_stagecap_privileged.yaml",
        "env_configs/two_arm_handover/two_arm_handover_stagecap_privileged.yaml",
    ),
)
def test_robosuite_configs_disable_libero_finalizer(config_path):
    _factory, config, _servers = _load_config(
        LaunchArgs(
            config_path=config_path,
            total_trials=None,
            num_workers=None,
            record_video=None,
            use_oracle_code=None,
        )
    )

    assert config["use_stage_verifiable_runner"]
    assert config["use_deterministic_stage_fast_paths"]
    assert config["disable_privileged_data_generation_finalize"]


def test_experiment_scripts_do_not_contain_plaintext_api_keys():
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/run_robosuite_comparison_tmux.sh",
        "scripts/run_libero40_kite_tmux.sh",
    ):
        assert "sk-" not in (root / relative_path).read_text(encoding="utf-8")


def test_batch_report_scans_runner_model_subdirectory():
    assert _model_output_dir(Path("outputs/run/task/run"), "Qwen/Qwen3.5-4B") == Path(
        "outputs/run/task/Qwen_Qwen3.5-4B/run"
    )


def test_batch_stagecap_model_toggle_does_not_change_agent0_baseline(tmp_path):
    base = {"record_video": True, "use_stage_verifiable_runner": True}
    agent0 = _build_config(
        base,
        RoboSuiteBatchArgs(condition="agent0", deterministic_fast_paths=True),
        tmp_path / "agent0",
    )
    stagecap_model = _build_config(
        base,
        RoboSuiteBatchArgs(condition="stagecap", deterministic_fast_paths=False),
        tmp_path / "stagecap_model",
    )
    stagecap_default = _build_config(
        base,
        RoboSuiteBatchArgs(condition="stagecap", deterministic_fast_paths=True),
        tmp_path / "stagecap_default",
    )

    assert not agent0["use_stage_verifiable_runner"]
    assert not agent0["use_deterministic_stage_fast_paths"]
    assert stagecap_model["use_stage_verifiable_runner"]
    assert not stagecap_model["use_deterministic_stage_fast_paths"]
    assert stagecap_default["use_stage_verifiable_runner"]
    assert stagecap_default["use_deterministic_stage_fast_paths"]


@pytest.mark.parametrize("task", ROBOSUITE_TASKS, ids=lambda task: task["name"])
def test_batch_agent0_uses_privileged_non_stagecap_config(task):
    agent0_path = Path(task["agent0_config"])
    stagecap_path = Path(task["stagecap_config"])
    agent0 = _load_yaml(agent0_path)
    stagecap = _load_yaml(stagecap_path)

    agent0_apis = tuple(agent0["env"]["cfg"].get("apis", ()))
    stagecap_apis = tuple(stagecap["env"]["cfg"].get("apis", ()))

    assert agent0_path != stagecap_path
    assert agent0["env"]["cfg"].get("privileged") is True
    assert any(name.endswith("PrivilegedApi") for name in agent0_apis)
    assert not any("Robosuite" in name and "StageApi" in name for name in agent0_apis)
    assert any("Robosuite" in name and "StageApi" in name for name in stagecap_apis)
    assert not agent0["env"]["cfg"].get("clean_code_guard", False)
    assert stagecap["env"]["cfg"].get("clean_code_guard", False)


def test_batch_rejects_stagecap_api_in_agent0_baseline():
    config = {
        "env": {
            "cfg": {
                "privileged": True,
                "apis": ["FrankaRobosuiteStackStageApi"],
            }
        }
    }

    with pytest.raises(ValueError, match="Agent0 must not receive StageCaP APIs"):
        _validate_condition_config(config, "agent0", "stagecap.yaml")


def test_batch_rejects_non_privileged_agent0_baseline():
    config = {
        "env": {
            "cfg": {
                "privileged": False,
                "apis": ["FrankaControlPrivilegedApi"],
            }
        }
    }

    with pytest.raises(ValueError, match="requires privileged: true"):
        _validate_condition_config(config, "agent0", "agent0.yaml")
