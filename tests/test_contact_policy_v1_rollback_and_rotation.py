"""Two targeted properties of contact policy v1.

1. Rollback isolation: the per-contact GEOMETRY statistics must be rolled back together with the contact events.
   A branch that produced a violating contact and was then discarded must leave no trace in the kept trajectory's
   verdict.  Discriminative form: run a dirty branch, restore, run a clean branch -> everything the final verdict
   reads must equal a direct run of the clean branch alone.
2. Rotation consistency: the inward closing axis is a world-frame direction.  The same physical inner-face contact,
   observed after the whole hand has rotated, must receive the same verdict.
"""
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402
from synthesis_v2.contact_policy_v1 import DetailRecorder, evaluate  # noqa: E402

# A yaw hinge carries both fingers AND the object, so rotating the hinge leaves the contact unchanged in the hand's
# own frame while turning the world-frame closing axis.  Finger boxes 0.01 half-thickness, inner faces at +-0.015.
XML = """
<mujoco>
  <worldbody>
    <body name="gripper0_hand" pos="0 0 0">
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom name="anchor" type="sphere" size="0.001" contype="0" conaffinity="0"/>
      <body name="gripper0_leftfinger" pos="-0.025 0 0">
        <geom name="gripper0_finger1_collision" type="box" size="0.01 0.02 0.03"/>
      </body>
      <body name="gripper0_rightfinger" pos="0.025 0 0">
        <geom name="gripper0_finger2_collision" type="box" size="0.01 0.02 0.03"/>
      </body>
    </body>
    <body name="cube_1_main" pos="0 0 0">
      <freejoint/>
      <geom name="cube" type="box" size="0.0152 0.02 0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


class _Sess:
    """Minimal session: one MuJoCo model, a settable yaw, and the save/restore pair the recorder wraps."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(XML)
        self.d = mujoco.MjData(self.m)
        self.control_step_index = 0
        self.force_target_name = "cube_1"
        self.env = SimpleNamespace(sim=SimpleNamespace(step=self._sub))

    def _sub(self):
        mujoco.mj_forward(self.m, self.d)

    def _mj(self):
        return self.m, self.d

    def set_yaw(self, rad):
        # the object turns with the hand, so the contact is unchanged in the hand's own frame and only the
        # world-frame closing axis moves -- which is exactly what a cached axis would get wrong
        self.d.qpos[0] = rad
        self.d.qpos[1:4] = 0.0
        self.d.qpos[4:8] = [np.cos(rad / 2), 0.0, 0.0, np.sin(rad / 2)]
        mujoco.mj_forward(self.m, self.d)

    def step(self, _=None):
        self.env.sim.step()
        self.control_step_index += 1

    def save_state(self):
        return {"c": self.control_step_index, "q": self.d.qpos.copy()}

    def restore_state(self, st):
        self.control_step_index = st["c"]
        self.d.qpos[:] = st["q"]
        mujoco.mj_forward(self.m, self.d)


class _Rec(ContactRecorder):
    """_scan() reads the real contacts, plus optional synthetic ones to inject a palm violation."""

    inject = ()

    def _scan(self):
        yield from super()._scan()
        yield from self.inject


def _mk():
    s = _Sess()
    r = _Rec(s)
    r.m, r.d = s.m, s.d
    r.install()
    det = DetailRecorder(r)
    return s, r, det


WINDOWS = [{"instance_id": "cube.pick_1.grasp", "object": "cube_1_main", "env_object": "cube_1",
            "close_step": 0, "release_step": 50}]


def _verdict(r, det, stages):
    rep = r.report(stages)
    return evaluate(rep, r.events, det.export(), WINDOWS, stages)


def _clean_branch(s, r):
    r.inject = ()
    for _ in range(3):
        s.step()


def _dirty_branch(s, r):
    r.inject = (("gripper0_right_gripper", "other_1_main", 9.0, -1, -1),)
    for _ in range(3):
        s.step()


def test_rolled_back_branch_leaves_no_trace_in_the_kept_verdict():
    s, r, det = _mk()
    base = s.save_state()
    _dirty_branch(s, r)
    s.restore_state(base)
    _clean_branch(s, r)
    stages = ["grasp"] * s.control_step_index
    got_detail, got = det.export(), _verdict(r, det, stages)

    s2, r2, det2 = _mk()
    _clean_branch(s2, r2)
    want_detail, want = det2.export(), _verdict(r2, det2, ["grasp"] * s2.control_step_index)

    assert r.events.keys() == r2.events.keys()
    assert set(got_detail) == set(want_detail), "discarded branch leaked a contact geometry row"
    for k in want_detail:
        assert got_detail[k]["n_substeps"] == want_detail[k]["n_substeps"], f"substep count polluted for {k}"
        assert got_detail[k]["n_substeps_inner_region"] == want_detail[k]["n_substeps_inner_region"]
        assert got_detail[k]["peak_n"] == pytest.approx(want_detail[k]["peak_n"])
    assert got["clean_v1"] == want["clean_v1"]
    assert [c["v1_excused"] for c in got["contacts"]] == [c["v1_excused"] for c in want["contacts"]]
    assert det.discarded_report()["branches"] == 1, "the discarded branch's cost must still be reported"


def test_inner_face_verdict_is_invariant_to_hand_rotation():
    s, r, det = _mk()
    _clean_branch(s, r)                       # contacts observed at yaw = 0
    at_zero = {k: v["n_substeps_inner_region"] for k, v in det.export().items()}
    assert at_zero and all(v > 0 for v in at_zero.values()), "setup must produce inner-face contacts at yaw 0"

    s.set_yaw(np.pi / 2)                      # same contact, hand turned 90 deg: world closing axis is now +y
    n_before = {k: v["n_substeps"] for k, v in det.export().items()}
    _clean_branch(s, r)
    after = det.export()
    for k, n0 in n_before.items():
        added = after[k]["n_substeps"] - n0
        added_inner = after[k]["n_substeps_inner_region"] - at_zero[k]
        assert added > 0
        assert added_inner == added, f"{k}: {added_inner}/{added} substeps judged inner after rotation"
