from synthesis_v2.program_executor import grasp_grid_from_params

GG = {"yaws_deg": [0, 90], "laterals_m": [0.0, 0.03], "dzs_m": [0.0, 0.02]}
GC = {"yaw_deg": 60.0, "lateral_m": 0.03, "dz_m": 0.08}


def test_default_when_nothing_given():
    assert grasp_grid_from_params({}) == (None, "default_grid")


def test_configured_grid_when_no_candidate():
    g, src = grasp_grid_from_params({"grasp_grid": GG})
    assert src == "grasp_grid" and len(g) == 8


def test_explicit_candidate_wins_over_grid():
    g, src = grasp_grid_from_params({"grasp_grid": GG, "grasp_candidate": GC})
    assert src == "grasp_candidate" and g == [(60.0, 0.03, 0.08, 0.0)]


def test_candidate_alone_unchanged():
    g, src = grasp_grid_from_params({"grasp_candidate": GC})
    assert src == "grasp_candidate" and g == [(60.0, 0.03, 0.08, 0.0)]


def test_reference_to_candidate_roundtrip_and_priority():
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from synthesis_v2.approach_planner import quat_to_mat, yaw_quat
    from synthesis_v2.program_executor import grasp_grid_from_params, reference_to_candidate
    down = R.from_euler("xyz", [np.pi, 0.1, 0.05]).as_quat()
    obs_to_site = np.eye(3)
    oq = R.from_euler("z", -np.pi / 2).as_quat()
    ref = {"rel_obj": [-0.03, -0.01, 0.1], "hand_yaw_rel_deg": 10.0}
    yaw, lat, dz, lon = reference_to_candidate(ref, oq, down, obs_to_site)
    Rq = quat_to_mat(yaw_quat(down, np.deg2rad(yaw)))
    off = Rq[:, 1] * lat + Rq[:, 0] * lon + np.array([0, 0, dz])
    want = R.from_quat(oq).apply(ref["rel_obj"])
    assert np.allclose(off, want, atol=1e-9)
    g, src = grasp_grid_from_params({"grasp_reference": ref, "grasp_grid": {"yaws_deg": [0], "laterals_m": [0.0], "dzs_m": [0.0]}}, (yaw, lat, dz, lon))
    assert src == "grasp_reference" and len(g) == 1
    g, src = grasp_grid_from_params({"grasp_candidate": {"yaw_deg": 3.0}, "grasp_reference": ref}, (yaw, lat, dz, lon))
    assert src == "grasp_candidate" and g[0][0] == 3.0
