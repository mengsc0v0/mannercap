# Stage code block 0
import json
import math

stage_id = 'middle_drawer_ground_handle'
level = 'middle'

try:
    handle_pos, hook_quat, pull_dir = get_cabinet_drawer_handle_pose(level)
    progress = get_cabinet_drawer_progress(level)
    hooks = get_drawer_sleeve_hook_poses(level)
    open_fraction = float(progress.get("open_fraction", 0.0) if isinstance(progress, dict) else progress)
    handle_ok = all(math.isfinite(float(x)) for x in handle_pos)
    pull_norm = math.sqrt(sum(float(x) * float(x) for x in pull_dir))
    hook_count = len(hooks.get("sleeve_candidates", [])) if isinstance(hooks, dict) else 0
    status = "satisfied" if handle_ok and abs(pull_norm - 1.0) <= 0.10 else "needs_repair"
    evidence = (
        f"level={level}, handle_ok={handle_ok}, pull_norm={pull_norm:.3f}, "
        f"open_fraction={open_fraction:.3f}, sleeve_candidates={hook_count}"
    )
    repair = "" if status == "satisfied" else "Drawer handle grounding failed; verify level and readonly drawer diagnostics."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer ground fast-path exception: {exc}"
    repair = "Retry drawer grounding with the correct drawer level."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}))


# Stage code block 1
import json

stage_id = 'middle_drawer_sleeve_or_hook_handle'
level = 'middle'
open_threshold = 0.98
require_container_clearance = False

try:
    progress = get_cabinet_drawer_progress(level)
    open_fraction = float(progress.get("open_fraction", 0.0) if isinstance(progress, dict) else progress)
    skip_threshold = open_threshold if require_container_clearance else 0.45
    if open_fraction >= skip_threshold:
        status = "satisfied"
        evidence = (
            f"level={level}, drawer_already_open=True, open_fraction={open_fraction:.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}"
        )
        repair = ""
    elif require_container_clearance and open_fraction >= 0.45:
        pull = open_drawer_by_sleeve_retry(
            level,
            open_threshold=open_threshold,
            max_cycles=3,
            test_distance=0.035,
            pull_distance=0.18,
            strict_open_fraction=True,
        )
        after = get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
        best = float(pull.get("best_open_fraction", after_fraction) if isinstance(pull, dict) else after_fraction)
        opened = bool(isinstance(pull, dict) and pull.get("opened", False))
        success = bool(opened or after_fraction >= open_threshold or best >= open_threshold)
        status = "satisfied" if success else "needs_repair"
        evidence = (
            f"level={level}, partial_drawer_open=True, before={open_fraction:.3f}, "
            f"after={after_fraction:.3f}, best={best:.3f}, opened={opened}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, result={pull}"
        )
        repair = "" if success else "Drawer is only partially open; continue opening to create verified physical clearance before proceeding."
    else:
        plan = get_drawer_sleeve_hook_poses(level)
        quat = plan.get("hook_quaternion_wxyz")
        candidates = plan.get("sleeve_candidates", []) if isinstance(plan, dict) else []
        records = []
        open_gripper()
        prehook = plan.get("prehook_finger_position") if isinstance(plan, dict) else None
        mouth = plan.get("mouth_finger_position") if isinstance(plan, dict) else None
        if prehook is not None and quat is not None:
            records.append({"action": "prehook", "move": goto_finger_midpoint_checked(prehook, quat, tolerance=0.080)})
        if mouth is not None and quat is not None:
            records.append({"action": "mouth", "move": goto_finger_midpoint_checked(mouth, quat, tolerance=0.070)})
        best = None
        for idx, cand in enumerate(candidates[:7]):
            target = cand.get("finger_midpoint_position")
            cand_quat = cand.get("quaternion_wxyz", quat)
            if target is None or cand_quat is None:
                continue
            move = goto_finger_midpoint_checked(target, cand_quat, tolerance=0.070)
            align_open = measure_drawer_hook_alignment(level)
            ready = bool(align_open.get("ready_to_close") or align_open.get("loose_hook_geometry") or align_open.get("centered_between_fingers"))
            close_gripper()
            align_closed = measure_drawer_hook_alignment(level)
            contacts = measure_drawer_gripper_contacts(level)
            captured = bool(align_closed.get("closed_on_handle") or align_closed.get("likely_hooked") or contacts.get("has_handle_contact"))
            record = {
                "candidate": cand.get("name", str(idx)),
                "move_arrived": bool(move.get("arrived", False)),
                "ready": ready,
                "captured": captured,
                "alignment": align_closed,
                "contacts": contacts,
            }
            records.append(record)
            best = record
            if captured:
                break
            open_gripper()
        captured = bool(best and best.get("captured"))
        status = "satisfied" if captured else "needs_repair"
        evidence = f"level={level}, captured={captured}, attempts={len(records)}, best={best}"
        repair = "" if captured else "Handle was not captured; retry sleeve/hook with another candidate."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer sleeve fast-path exception: {exc}"
    repair = "Retry drawer handle capture from prehook/mouth pose."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 2
import json

stage_id = 'middle_drawer_test_progress'
level = 'middle'
open_threshold = 0.98
require_container_clearance = False

try:
    before = get_cabinet_drawer_progress(level)
    before_fraction = float(before.get("open_fraction", 0.0) if isinstance(before, dict) else before)
    before_bddl_open = bool(isinstance(before, dict) and before.get("bddl_open", False))
    already_open_enough = (
        bool(before_fraction >= open_threshold)
        if require_container_clearance
        else bool(before_bddl_open or before_fraction >= open_threshold)
    )
    if already_open_enough:
        status = "satisfied"
        evidence = (
            f"level={level}, drawer_already_open=True, open_fraction={before_fraction:.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, progress={before}"
        )
        repair = ""
    else:
        pull = open_drawer_by_sleeve_retry(
            level,
            open_threshold=open_threshold,
            max_cycles=5 if require_container_clearance else 3,
            test_distance=0.055,
            pull_distance=0.48 if require_container_clearance else (0.34 if open_threshold < 0.90 else 0.42),
            strict_open_fraction=bool(require_container_clearance),
        )
        after = get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
        delta = after_fraction - before_fraction
        contacts = measure_drawer_gripper_contacts(level)
        alignment = measure_drawer_hook_alignment(level)
        best = float(pull.get("best_open_fraction", after_fraction) if isinstance(pull, dict) else after_fraction)
        opened = bool(isinstance(pull, dict) and pull.get("opened", False))
        progressed = (
            bool(opened or best >= open_threshold or after_fraction >= open_threshold)
            if require_container_clearance
            else bool(opened or best >= open_threshold or delta >= 0.015 or after_fraction >= 0.12)
        )
        status = "satisfied" if progressed else "needs_repair"
        evidence = (
            f"level={level}, before={before_fraction:.3f}, after={after_fraction:.3f}, "
            f"delta={delta:.3f}, best={best:.3f}, opened={opened}, "
            f"bddl_open={bool(isinstance(after, dict) and after.get('bddl_open', False))}, "
            f"bddl_margin={float(after.get('bddl_open_margin', 0.0) if isinstance(after, dict) else 0.0):.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, "
            f"handle_contact={bool(contacts.get('has_handle_contact', False))}, "
            f"closed_on_handle={bool(alignment.get('closed_on_handle', False))}, result={pull}"
        )
        repair = "" if progressed else "Drawer is not physically open enough for the downstream drawer objective; keep the handle captured and pull farther before proceeding."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer test fast-path exception: {exc}"
    repair = "Retry drawer progress test after handle capture."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 3
import json

stage_id = 'middle_drawer_test_progress'
level = 'middle'
open_threshold = 0.98
require_container_clearance = False

try:
    before = get_cabinet_drawer_progress(level)
    before_fraction = float(before.get("open_fraction", 0.0) if isinstance(before, dict) else before)
    before_bddl_open = bool(isinstance(before, dict) and before.get("bddl_open", False))
    already_open_enough = (
        bool(before_fraction >= open_threshold)
        if require_container_clearance
        else bool(before_bddl_open or before_fraction >= open_threshold)
    )
    if already_open_enough:
        status = "satisfied"
        evidence = (
            f"level={level}, drawer_already_open=True, open_fraction={before_fraction:.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, progress={before}"
        )
        repair = ""
    else:
        pull = open_drawer_by_sleeve_retry(
            level,
            open_threshold=open_threshold,
            max_cycles=5 if require_container_clearance else 3,
            test_distance=0.055,
            pull_distance=0.48 if require_container_clearance else (0.34 if open_threshold < 0.90 else 0.42),
            strict_open_fraction=bool(require_container_clearance),
        )
        after = get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
        delta = after_fraction - before_fraction
        contacts = measure_drawer_gripper_contacts(level)
        alignment = measure_drawer_hook_alignment(level)
        best = float(pull.get("best_open_fraction", after_fraction) if isinstance(pull, dict) else after_fraction)
        opened = bool(isinstance(pull, dict) and pull.get("opened", False))
        progressed = (
            bool(opened or best >= open_threshold or after_fraction >= open_threshold)
            if require_container_clearance
            else bool(opened or best >= open_threshold or delta >= 0.015 or after_fraction >= 0.12)
        )
        status = "satisfied" if progressed else "needs_repair"
        evidence = (
            f"level={level}, before={before_fraction:.3f}, after={after_fraction:.3f}, "
            f"delta={delta:.3f}, best={best:.3f}, opened={opened}, "
            f"bddl_open={bool(isinstance(after, dict) and after.get('bddl_open', False))}, "
            f"bddl_margin={float(after.get('bddl_open_margin', 0.0) if isinstance(after, dict) else 0.0):.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, "
            f"handle_contact={bool(contacts.get('has_handle_contact', False))}, "
            f"closed_on_handle={bool(alignment.get('closed_on_handle', False))}, result={pull}"
        )
        repair = "" if progressed else "Drawer is not physically open enough for the downstream drawer objective; keep the handle captured and pull farther before proceeding."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer test fast-path exception: {exc}"
    repair = "Retry drawer progress test after handle capture."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 4
import json

stage_id = 'middle_drawer_test_progress'
level = 'middle'
open_threshold = 0.98
require_container_clearance = False

try:
    before = get_cabinet_drawer_progress(level)
    before_fraction = float(before.get("open_fraction", 0.0) if isinstance(before, dict) else before)
    before_bddl_open = bool(isinstance(before, dict) and before.get("bddl_open", False))
    already_open_enough = (
        bool(before_fraction >= open_threshold)
        if require_container_clearance
        else bool(before_bddl_open or before_fraction >= open_threshold)
    )
    if already_open_enough:
        status = "satisfied"
        evidence = (
            f"level={level}, drawer_already_open=True, open_fraction={before_fraction:.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, progress={before}"
        )
        repair = ""
    else:
        pull = open_drawer_by_sleeve_retry(
            level,
            open_threshold=open_threshold,
            max_cycles=5 if require_container_clearance else 3,
            test_distance=0.055,
            pull_distance=0.48 if require_container_clearance else (0.34 if open_threshold < 0.90 else 0.42),
            strict_open_fraction=bool(require_container_clearance),
        )
        after = get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
        delta = after_fraction - before_fraction
        contacts = measure_drawer_gripper_contacts(level)
        alignment = measure_drawer_hook_alignment(level)
        best = float(pull.get("best_open_fraction", after_fraction) if isinstance(pull, dict) else after_fraction)
        opened = bool(isinstance(pull, dict) and pull.get("opened", False))
        progressed = (
            bool(opened or best >= open_threshold or after_fraction >= open_threshold)
            if require_container_clearance
            else bool(opened or best >= open_threshold or delta >= 0.015 or after_fraction >= 0.12)
        )
        status = "satisfied" if progressed else "needs_repair"
        evidence = (
            f"level={level}, before={before_fraction:.3f}, after={after_fraction:.3f}, "
            f"delta={delta:.3f}, best={best:.3f}, opened={opened}, "
            f"bddl_open={bool(isinstance(after, dict) and after.get('bddl_open', False))}, "
            f"bddl_margin={float(after.get('bddl_open_margin', 0.0) if isinstance(after, dict) else 0.0):.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, "
            f"handle_contact={bool(contacts.get('has_handle_contact', False))}, "
            f"closed_on_handle={bool(alignment.get('closed_on_handle', False))}, result={pull}"
        )
        repair = "" if progressed else "Drawer is not physically open enough for the downstream drawer objective; keep the handle captured and pull farther before proceeding."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer test fast-path exception: {exc}"
    repair = "Retry drawer progress test after handle capture."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 5
import json

stage_id = 'middle_drawer_test_progress'
level = 'middle'
open_threshold = 0.98
require_container_clearance = False

try:
    before = get_cabinet_drawer_progress(level)
    before_fraction = float(before.get("open_fraction", 0.0) if isinstance(before, dict) else before)
    before_bddl_open = bool(isinstance(before, dict) and before.get("bddl_open", False))
    already_open_enough = (
        bool(before_fraction >= open_threshold)
        if require_container_clearance
        else bool(before_bddl_open or before_fraction >= open_threshold)
    )
    if already_open_enough:
        status = "satisfied"
        evidence = (
            f"level={level}, drawer_already_open=True, open_fraction={before_fraction:.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, progress={before}"
        )
        repair = ""
    else:
        pull = open_drawer_by_sleeve_retry(
            level,
            open_threshold=open_threshold,
            max_cycles=5 if require_container_clearance else 3,
            test_distance=0.055,
            pull_distance=0.48 if require_container_clearance else (0.34 if open_threshold < 0.90 else 0.42),
            strict_open_fraction=bool(require_container_clearance),
        )
        after = get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
        delta = after_fraction - before_fraction
        contacts = measure_drawer_gripper_contacts(level)
        alignment = measure_drawer_hook_alignment(level)
        best = float(pull.get("best_open_fraction", after_fraction) if isinstance(pull, dict) else after_fraction)
        opened = bool(isinstance(pull, dict) and pull.get("opened", False))
        progressed = (
            bool(opened or best >= open_threshold or after_fraction >= open_threshold)
            if require_container_clearance
            else bool(opened or best >= open_threshold or delta >= 0.015 or after_fraction >= 0.12)
        )
        status = "satisfied" if progressed else "needs_repair"
        evidence = (
            f"level={level}, before={before_fraction:.3f}, after={after_fraction:.3f}, "
            f"delta={delta:.3f}, best={best:.3f}, opened={opened}, "
            f"bddl_open={bool(isinstance(after, dict) and after.get('bddl_open', False))}, "
            f"bddl_margin={float(after.get('bddl_open_margin', 0.0) if isinstance(after, dict) else 0.0):.3f}, "
            f"open_threshold={open_threshold:.3f}, require_container_clearance={require_container_clearance}, "
            f"handle_contact={bool(contacts.get('has_handle_contact', False))}, "
            f"closed_on_handle={bool(alignment.get('closed_on_handle', False))}, result={pull}"
        )
        repair = "" if progressed else "Drawer is not physically open enough for the downstream drawer objective; keep the handle captured and pull farther before proceeding."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer test fast-path exception: {exc}"
    repair = "Retry drawer progress test after handle capture."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))
