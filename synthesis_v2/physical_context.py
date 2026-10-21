"""Physical-context overlay: change the grasped object's mass / friction / size after reset.

The overlay is applied by ``ContextSession`` (a ``NativeSession``) inside ``reset()`` and the observation and the
recorded initial state are refreshed, so everything downstream (teacher, audits) sees the modified world.  A context
of all-``None`` fields is an exact no-op (the trajectory is bit-identical to the plain session).

Friction note: MuJoCo combines the contact friction of two geoms with max().  The Panda finger pads have mu = 2.0, so
lowering the object's friction alone has no effect.  When ``friction`` is set the object's geom gets a higher
``geom_priority`` so that ITS friction is used for the contact (its solref/solimp/condim apply as well; they are the
robosuite defaults).  ``friction`` is therefore the effective sliding friction of the finger-object contact.

Only the object the gripper grasps is modified (Lift: ``cube``; Stack: ``cubeA``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from synthesis_v2.native_osc_env import NativeSession

CONTEXT_VERSION = "physical_context.v1"
GRASPED_TASKS = ("lift", "stack")


@dataclass(frozen=True)
class PhysicalContext:
    mass_kg: Optional[float] = None       # total object mass; None keeps the robosuite value (density x volume)
    friction: Optional[float] = None      # effective sliding friction of finger-object contact
    size_scale: Optional[float] = None    # isotropic scale of the object's half extents

    def is_noop(self) -> bool:
        return self.mass_kg is None and self.friction is None and self.size_scale is None

    def validate(self) -> None:
        if self.mass_kg is not None and not 0.005 <= self.mass_kg <= 3.0:
            raise ValueError(f"mass_kg {self.mass_kg} outside [0.005, 3.0]")
        if self.friction is not None and not 0.02 <= self.friction <= 3.0:
            raise ValueError(f"friction {self.friction} outside [0.02, 3.0]")
        if self.size_scale is not None and not 0.5 <= self.size_scale <= 1.6:
            raise ValueError(f"size_scale {self.size_scale} outside [0.5, 1.6]")

    def as_dict(self) -> dict[str, Any]:
        return {"version": CONTEXT_VERSION, **asdict(self)}


class ContextSession(NativeSession):
    """NativeSession whose reset() applies a PhysicalContext to the grasped object."""

    def __init__(self, task: str, seed: int, context: PhysicalContext, *, render: bool = True) -> None:
        if task not in GRASPED_TASKS and not context.is_noop():
            raise ValueError(f"physical context is defined for {GRASPED_TASKS}, not {task!r}")
        context.validate()
        super().__init__(task, seed, render=render)
        self.context = context
        self.context_record: dict[str, Any] = {"context": context.as_dict(), "applied": False}

    def reset(self) -> dict:
        super().reset()
        if self.context.is_noop():
            return self.obs
        import mujoco

        m, d = self._mj()
        attr, geoms = self._target_geoms()
        if not geoms:
            raise RuntimeError("no target object geoms found")
        body = int(m.geom_bodyid[geoms[0]])
        old_mass = float(m.body_mass[body])
        old_half_z = float(m.geom_size[geoms[0]][2])
        scale = float(self.context.size_scale or 1.0)
        mass_new = float(self.context.mass_kg) if self.context.mass_kg is not None else old_mass * scale ** 3
        if scale != 1.0:
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) == body:
                    m.geom_size[g] = m.geom_size[g] * scale
                    m.geom_rbound[g] = m.geom_rbound[g] * scale
        m.body_inertia[body] = m.body_inertia[body] * (mass_new / old_mass) * scale ** 2
        m.body_mass[body] = mass_new
        if self.context.friction is not None:
            for g in geoms:
                m.geom_friction[g][0] = float(self.context.friction)
                m.geom_priority[g] = 1
        if scale != 1.0:
            jnt = int(m.body_jntadr[body])
            adr = int(m.jnt_qposadr[jnt])
            d.qpos[adr + 2] += (scale - 1.0) * old_half_z  # keep the object resting on the table
        mujoco.mj_forward(m, d)
        self.obs = self.env._get_observations(force_update=True)
        self.initial = self.snapshot_initial()
        self.context_record = {
            "context": self.context.as_dict(), "applied": True, "object": attr,
            "mass_kg": float(m.body_mass[body]), "mass_orig_kg": old_mass,
            "friction": float(m.geom_friction[geoms[0]][0]), "half_extents_m": m.geom_size[geoms[0]].tolist(),
        }
        return self.obs
