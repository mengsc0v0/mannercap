"""Grasp-closure policy (default unchanged; settle / bounded re-close are opt-in) and the post-hoc audit flags."""
from synthesis_v2.grasp_audit import grasp_flags
from synthesis_v2.native_osc_env import StepTelemetry
from synthesis_v2.native_producer import GRIP_CLOSE, GRIP_HOLD, GRIP_OPEN, ContactFeedbackGrip, GripPlan


def tel(left, right):
    return StepTelemetry(n_substeps=10, left_mean_n=left, right_mean_n=right, left_max_n=left, right_max_n=right,
                         other_max_n=0.0, valid_fraction=1.0, joint_speed_mean=0.0, joint_speed_max=0.0)


def drive(plan, readings):
    pol = ContactFeedbackGrip(plan)
    cmds = [pol.command(None)]
    for lf, rt in readings:
        cmds.append(pol.command(tel(lf, rt)))
    return pol, cmds


def test_default_policy_unchanged():
    pol, cmds = drive(GripPlan(3.0, 9.0), [(1.0, 0.0), (2.0, 0.0), (8.0, 2.4), (4.0, 0.0), (2.0, 0.0)])
    assert cmds == [GRIP_CLOSE, GRIP_CLOSE, GRIP_CLOSE, GRIP_OPEN, GRIP_HOLD, GRIP_HOLD]
    assert pol.reversed_pulses == 1 and pol.reclose_pulses == 0


def test_settle_waits_one_reading_after_release():
    pol, cmds = drive(GripPlan(3.0, 9.0, settle_after_open=1), [(10.0, 2.0), (10.0, 2.0), (4.0, 1.0)])
    assert cmds[1] == GRIP_OPEN and cmds[2] == GRIP_HOLD and cmds[3] == GRIP_HOLD
    assert pol.reversed_pulses == 1


def test_reclose_is_bounded_and_only_when_below_floor():
    plan = GripPlan(3.0, 9.0, settle_after_open=1, reclose_max=2)
    pol, cmds = drive(plan, [(6.0, 0.0), (10.0, 2.0), (6.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
    assert cmds.count(GRIP_CLOSE) >= 3 and pol.reclose_pulses <= 2


def test_audit_flags_separate_history_from_current_contact():
    info = {"reached_band": True, "reverse_pulses": 2,
            "trace": [{"left_n": 5.0, "right_n": 4.0}, {"left_n": 3.0, "right_n": 1.0}],
            "final": {"left_n": 0.0, "right_n": 0.0, "total_n": 0.0}}
    f = grasp_flags(info)
    assert f["ever_reached_band"] and not f["contact_at_end"] and not f["prelift_stable"]
    info["final"] = {"left_n": 1.0, "right_n": 1.2, "total_n": 2.2}
    f = grasp_flags(info)
    assert f["contact_at_end"] and f["prelift_stable"]
