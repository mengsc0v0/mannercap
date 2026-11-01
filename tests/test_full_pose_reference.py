"""Step 23: grasp reference with a full hand orientation (tilt kept) vs the unchanged yaw-only reference."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from synthesis_v2.approach_planner import quat_to_mat, yaw_quat
from synthesis_v2.program_executor import reference_to_candidate

DOWN = R.from_euler("xyz", [np.pi, 0, 0]).as_quat()
EYE = np.eye(3)
OBJ_Q = R.from_euler("z", 0.4).as_quat()


def _legacy(ref, obj_q, grasp_q, o2s):   # the implementation before Step 23, verbatim
    oyaw = R.from_quat(obj_q).as_euler("xyz")[2]
    off_w = R.from_euler("z", oyaw).apply(np.asarray(ref["rel_obj"], float))
    yaw = float(ref["hand_yaw_rel_deg"])
    Rq = quat_to_mat(yaw_quat(np.asarray(grasp_q, float), np.deg2rad(yaw))) @ np.asarray(o2s, float)
    lon, lat, dz = np.linalg.solve(np.stack([Rq[:, 0], Rq[:, 1], [0.0, 0.0, 1.0]], axis=1), off_w)
    return (yaw, float(lat), float(dz), float(lon))


def _world_off(c, o2s):
    Rq = quat_to_mat(c[4] if len(c) > 4 else yaw_quat(DOWN, np.deg2rad(c[0]))) @ o2s
    return Rq[:, 1] * c[1] + Rq[:, 0] * c[3] + np.array([0, 0, c[2]])


def test_reference_without_full_pose_is_unchanged():
    ref = {"rel_obj": [0.01, -0.02, 0.1], "hand_yaw_rel_deg": 12.0}
    assert reference_to_candidate(ref, OBJ_Q, DOWN, EYE) == _legacy(ref, OBJ_Q, DOWN, EYE)
    assert len(reference_to_candidate(ref, OBJ_Q, DOWN, EYE)) == 4


def test_untilted_full_pose_gives_the_yaw_only_offset():
    ref = {"rel_obj": [0.01, -0.02, 0.1], "hand_yaw_rel_deg": 12.0}
    # a vertical hand turned 12 deg about z from the executor's yaw-aligned quaternion (object yaw 0.4 rad), expressed in
    # the object yaw frame: Rz(0.4)^-1 * Rz(12 deg) * Rz(0.4) * DOWN = Rz(12 deg) * DOWN
    q_obj = (R.from_euler("z", np.deg2rad(12.0)) * R.from_quat(DOWN)).as_quat()
    grasp_q = (R.from_euler("z", 0.4) * R.from_quat(DOWN)).as_quat()
    full = reference_to_candidate({**ref, "hand_quat_obj_xyzw": q_obj.tolist()}, OBJ_Q, grasp_q, EYE)
    old = reference_to_candidate(ref, OBJ_Q, grasp_q, EYE)
    assert np.allclose(full[:4], old, atol=1e-9)
    assert np.allclose(R.from_quat(full[4]).as_matrix(),
                       (R.from_euler("z", np.deg2rad(12.0)) * R.from_quat(grasp_q)).as_matrix(), atol=1e-9)


def test_tilt_is_kept_and_follows_the_object_yaw():
    tilt = R.from_euler("x", np.deg2rad(25.0)) * R.from_quat(DOWN)
    ref = {"rel_obj": [0.02, 0.03, 0.05], "hand_yaw_rel_deg": 0.0, "hand_quat_obj_xyzw": tilt.as_quat().tolist()}
    c0 = reference_to_candidate(ref, R.identity().as_quat(), DOWN, EYE)
    c1 = reference_to_candidate(ref, R.from_euler("z", 0.7).as_quat(), DOWN, EYE)
    t = lambda q: float(np.degrees(np.arccos(np.clip(-R.from_quat(q).apply([0, 0, 1])[2], -1, 1))))  # noqa: E731
    assert abs(t(c0[4]) - 25.0) < 1e-6 and abs(t(c1[4]) - 25.0) < 1e-6
    # object turned by 0.7 rad: hand orientation and hand offset are both turned by the same rotation about z
    assert np.allclose(R.from_quat(c1[4]).as_matrix(), R.from_euler("z", 0.7).as_matrix() @ R.from_quat(c0[4]).as_matrix(), atol=1e-9)
    assert np.allclose(_world_off(c1, EYE), R.from_euler("z", 0.7).apply(_world_off(c0, EYE)), atol=1e-9)
    assert np.allclose(_world_off(c0, EYE), R.from_euler("z", 0.0).apply(ref["rel_obj"]), atol=1e-9)
