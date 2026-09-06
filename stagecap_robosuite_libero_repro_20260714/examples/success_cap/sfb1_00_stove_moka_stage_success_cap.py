# Stage code block 0
STAGE_ID = "stove_ground_button"
STAGE_ATTEMPT = 1
import json

stage_id = 'stove_ground_button'
try:
    state = get_object_state_diagnostic("flat_stove_1")
    button_pos, button_quat = get_body_pose("flat_stove_1_button")
    status = "satisfied"
    evidence = f"stove_state={state}, button_pos={button_pos}, button_quat={button_quat}"
    repair = ""
except Exception as exc:
    status = "needs_repair"
    evidence = f"stove grounding exception: {exc}"
    repair = "Verify flat_stove and button body names before pressing."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 1
STAGE_ID = "stove_press_and_verify"
STAGE_ATTEMPT = 1
import json

stage_id = 'stove_press_and_verify'
try:
    result = turn_on_stove_by_sweep()
    status_info = get_task_status()
    reward = float(status_info.get("reward", 0.0) or 0.0)
    completed = bool(status_info.get("task_completed", False))
    turned_on = bool(isinstance(result, dict) and result.get("turned_on", False))
    status = "satisfied" if turned_on or completed or reward >= 1.0 else "needs_repair"
    evidence = f"turned_on={turned_on}, reward={reward:.3f}, task_completed={completed}, result={result}"
    repair = "" if status == "satisfied" else "Stove sweep did not turn on the stove; retry with alternate contact sweep."
except Exception as exc:
    status = "needs_repair"
    evidence = f"stove press fast-path exception: {exc}"
    repair = "Retry stove button sweep after checking button grounding."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 2
STAGE_ID = "obj1_moka_pot_pick_lift_verify"
STAGE_ATTEMPT = 1
import json

stage_id = 'obj1_moka_pot_pick_lift_verify'
object_name = 'moka_pot'
target_name = 'moka_pot'
predicate = 'on'
candidate_index = 0
drawer_bowl_pick = False
drawer_level = None
fragile_support_grasp = bool(
    predicate == "on"
    and (
        any(token in str(object_name).lower() for token in ("mug", "cup", "bowl", "ramekin"))
        or any(token in str(target_name).lower() for token in ("plate", "bowl"))
    )
)

print(f"Stage: {stage_id} - deterministic audited pick-lift for {object_name} candidate_index={candidate_index}")
try:
    drawer_preopen = None
    if drawer_bowl_pick:
        drawer_required_open_fraction = 0.96 if str(drawer_level or "").lower() == "bottom" else 0.98
        drawer_before_guard = get_cabinet_drawer_progress(drawer_level or "middle")
        drawer_before_fraction = float(drawer_before_guard.get("open_fraction", 0.0) if isinstance(drawer_before_guard, dict) else drawer_before_guard)
        drawer_before_bddl_open = bool(isinstance(drawer_before_guard, dict) and drawer_before_guard.get("bddl_open", False))
        if drawer_before_fraction >= drawer_required_open_fraction:
            drawer_preopen = {"reason": "preserve_strictly_open_drawer_for_bowl_insertion", "after": drawer_before_guard, "opened": True}
        else:
            drawer_preopen = open_drawer_by_sleeve_retry(
                drawer_level or "middle",
                open_threshold=drawer_required_open_fraction,
                max_cycles=5,
                test_distance=0.055,
                pull_distance=0.48,
                strict_open_fraction=True,
            )
        print("Drawer pre-pick open guard:", {"drawer_level": drawer_level, "before": drawer_before_guard, "result": drawer_preopen})
        drawer_after_guard = drawer_preopen.get("after", {}) if isinstance(drawer_preopen, dict) else {}
        drawer_after_fraction = float(drawer_after_guard.get("open_fraction", 0.0) or 0.0)
        drawer_after_bddl_open = bool(isinstance(drawer_after_guard, dict) and drawer_after_guard.get("bddl_open", False))
        if drawer_after_fraction < drawer_required_open_fraction:
            evidence = (
                f"drawer pre-pick open guard failed, level={drawer_level}, "
                f"open_fraction={drawer_after_fraction:.3f}, required={drawer_required_open_fraction:.3f}, "
                f"bddl_open={drawer_after_bddl_open}, result={drawer_preopen}"
            )
            raise Exception(evidence)
    before = measure_object_grasp_state(object_name)
    print("Before grasp:", before)
    object_text = str(object_name).lower()
    if drawer_bowl_pick:
        grasp = verified_low_side_pick_object(
            object_name,
            max_attempts=2,
            start_candidate_index=candidate_index,
            lift_height=0.075,
            settle_steps=18,
        )
        print("Drawer bowl low-side pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False):
            print("Drawer bowl low-side pick failed; falling back to geometry pick candidate.")
            grasp = verified_geometry_pick_object(
                object_name,
                max_attempts=1,
                start_candidate_index=max(0, candidate_index - 2),
                lift_height=0.22,
            )
            print("Drawer bowl geometry fallback result:", grasp)
    elif "moka" in object_text or "book" in object_text:
        grasp = verified_geometry_pick_object(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Geometry pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False) and "moka" in object_text:
            print("Moka geometry pick was not transport-stable; trying affordance grasp fallback.")
            grasp = approach_and_grasp_object(
                object_name,
                max_attempts=2,
                start_candidate_index=max(0, candidate_index % 6),
                lift_height=0.135,
            )
            print("Moka affordance fallback result:", grasp)
    else:
        direct_attempts = 3 if drawer_bowl_pick else 1
        grasp = verified_lift_pick_object(object_name, max_attempts=direct_attempts, start_candidate_index=candidate_index)
        print("Audited grasp result:", grasp)
    if isinstance(grasp, dict) and (not drawer_bowl_pick) and (("bowl" in str(object_name).lower()) or fragile_support_grasp):
        strict_after_direct = measure_object_grasp_state(grasp.get("object_name", object_name), strict=True)
        direct_lift = float(grasp.get("lift_delta", 0.0) or 0.0)
        direct_stable = bool(grasp.get("held", False) and (strict_after_direct.get("held", False) or direct_lift >= 0.060))
        if not direct_stable:
            print("Direct pick is not transport-stable for fragile support placement; trying affordance fallback.")
            fallback_index = max(0, candidate_index)
            grasp = approach_and_grasp_object(object_name, max_attempts=2, start_candidate_index=fallback_index)
            print("Fragile support affordance fallback result:", grasp)
            if isinstance(grasp, dict) and not grasp.get("held", False) and ("mug" in str(object_name).lower()):
                print("Mug affordance fallback failed; trying side-grasp fallback.")
                grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
                print("Mug side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and ("wine" in str(object_name).lower() or "bottle" in str(object_name).lower()):
        print("Tall/narrow object direct pick failed; trying audited side-grasp fallback once.")
        grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Tall side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and candidate_index >= 11:
        print("Direct verified pick exhausted canonical offsets; trying affordance fallback once.")
        grasp = approach_and_grasp_object(object_name, max_attempts=1, start_candidate_index=max(0, candidate_index - 11))
        print("Affordance fallback result:", grasp)
    resolved_name = grasp.get("object_name", object_name) if isinstance(grasp, dict) else object_name
    after = measure_object_grasp_state(resolved_name)
    print("After grasp:", after)
    held = bool(isinstance(grasp, dict) and grasp.get("held")) or bool(isinstance(after, dict) and after.get("held"))
    lift_delta = 0.0
    if isinstance(grasp, dict):
        lift_delta = max(lift_delta, float(grasp.get("lift_delta", 0.0) or 0.0))
    if isinstance(after, dict):
        lift_delta = max(lift_delta, float(after.get("lift_delta", 0.0) or 0.0))
    stable_held = bool(held or lift_delta >= 0.035)
    strict_hold = None
    if fragile_support_grasp:
        strict_hold = measure_object_grasp_state(resolved_name, strict=True)
        stable_held = bool(held and (strict_hold.get("held", False) or lift_delta >= 0.060))
    if drawer_bowl_pick:
        transport_probe = grasp.get("transport_probe") if isinstance(grasp, dict) else None
        if not isinstance(transport_probe, dict) and isinstance(grasp, dict) and isinstance(grasp.get("best_attempt"), dict):
            transport_probe = grasp.get("best_attempt", {}).get("transport_probe")
        probe_held = bool(isinstance(transport_probe, dict) and transport_probe.get("held", False))
        stable_held = bool(held and (probe_held or lift_delta >= 0.070))
    status = "satisfied" if stable_held else "needs_repair"
    evidence = f"held={held}, stable_held={stable_held}, lift_delta={lift_delta:.4f}, object={resolved_name}, target={target_name}, candidate_index={candidate_index}, drawer_level={drawer_level}"
    repair = "" if status == "satisfied" else "Audited single grasp attempt failed; retry stage with next snapshot/candidate."
except Exception as exc:
    status = "needs_repair"
    evidence = f"audited pick-lift fast-path exception: {exc}"
    repair = "Retry stage if episode is still active; otherwise reset trial or reduce motion budget."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 3
STAGE_ID = "obj1_moka_pot_pick_lift_verify"
STAGE_ATTEMPT = 2
import json

stage_id = 'obj1_moka_pot_pick_lift_verify'
object_name = 'moka_pot'
target_name = 'moka_pot'
predicate = 'on'
candidate_index = 1
drawer_bowl_pick = False
drawer_level = None
fragile_support_grasp = bool(
    predicate == "on"
    and (
        any(token in str(object_name).lower() for token in ("mug", "cup", "bowl", "ramekin"))
        or any(token in str(target_name).lower() for token in ("plate", "bowl"))
    )
)

print(f"Stage: {stage_id} - deterministic audited pick-lift for {object_name} candidate_index={candidate_index}")
try:
    drawer_preopen = None
    if drawer_bowl_pick:
        drawer_required_open_fraction = 0.96 if str(drawer_level or "").lower() == "bottom" else 0.98
        drawer_before_guard = get_cabinet_drawer_progress(drawer_level or "middle")
        drawer_before_fraction = float(drawer_before_guard.get("open_fraction", 0.0) if isinstance(drawer_before_guard, dict) else drawer_before_guard)
        drawer_before_bddl_open = bool(isinstance(drawer_before_guard, dict) and drawer_before_guard.get("bddl_open", False))
        if drawer_before_fraction >= drawer_required_open_fraction:
            drawer_preopen = {"reason": "preserve_strictly_open_drawer_for_bowl_insertion", "after": drawer_before_guard, "opened": True}
        else:
            drawer_preopen = open_drawer_by_sleeve_retry(
                drawer_level or "middle",
                open_threshold=drawer_required_open_fraction,
                max_cycles=5,
                test_distance=0.055,
                pull_distance=0.48,
                strict_open_fraction=True,
            )
        print("Drawer pre-pick open guard:", {"drawer_level": drawer_level, "before": drawer_before_guard, "result": drawer_preopen})
        drawer_after_guard = drawer_preopen.get("after", {}) if isinstance(drawer_preopen, dict) else {}
        drawer_after_fraction = float(drawer_after_guard.get("open_fraction", 0.0) or 0.0)
        drawer_after_bddl_open = bool(isinstance(drawer_after_guard, dict) and drawer_after_guard.get("bddl_open", False))
        if drawer_after_fraction < drawer_required_open_fraction:
            evidence = (
                f"drawer pre-pick open guard failed, level={drawer_level}, "
                f"open_fraction={drawer_after_fraction:.3f}, required={drawer_required_open_fraction:.3f}, "
                f"bddl_open={drawer_after_bddl_open}, result={drawer_preopen}"
            )
            raise Exception(evidence)
    before = measure_object_grasp_state(object_name)
    print("Before grasp:", before)
    object_text = str(object_name).lower()
    if drawer_bowl_pick:
        grasp = verified_low_side_pick_object(
            object_name,
            max_attempts=2,
            start_candidate_index=candidate_index,
            lift_height=0.075,
            settle_steps=18,
        )
        print("Drawer bowl low-side pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False):
            print("Drawer bowl low-side pick failed; falling back to geometry pick candidate.")
            grasp = verified_geometry_pick_object(
                object_name,
                max_attempts=1,
                start_candidate_index=max(0, candidate_index - 2),
                lift_height=0.22,
            )
            print("Drawer bowl geometry fallback result:", grasp)
    elif "moka" in object_text or "book" in object_text:
        grasp = verified_geometry_pick_object(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Geometry pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False) and "moka" in object_text:
            print("Moka geometry pick was not transport-stable; trying affordance grasp fallback.")
            grasp = approach_and_grasp_object(
                object_name,
                max_attempts=2,
                start_candidate_index=max(0, candidate_index % 6),
                lift_height=0.135,
            )
            print("Moka affordance fallback result:", grasp)
    else:
        direct_attempts = 3 if drawer_bowl_pick else 1
        grasp = verified_lift_pick_object(object_name, max_attempts=direct_attempts, start_candidate_index=candidate_index)
        print("Audited grasp result:", grasp)
    if isinstance(grasp, dict) and (not drawer_bowl_pick) and (("bowl" in str(object_name).lower()) or fragile_support_grasp):
        strict_after_direct = measure_object_grasp_state(grasp.get("object_name", object_name), strict=True)
        direct_lift = float(grasp.get("lift_delta", 0.0) or 0.0)
        direct_stable = bool(grasp.get("held", False) and (strict_after_direct.get("held", False) or direct_lift >= 0.060))
        if not direct_stable:
            print("Direct pick is not transport-stable for fragile support placement; trying affordance fallback.")
            fallback_index = max(0, candidate_index)
            grasp = approach_and_grasp_object(object_name, max_attempts=2, start_candidate_index=fallback_index)
            print("Fragile support affordance fallback result:", grasp)
            if isinstance(grasp, dict) and not grasp.get("held", False) and ("mug" in str(object_name).lower()):
                print("Mug affordance fallback failed; trying side-grasp fallback.")
                grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
                print("Mug side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and ("wine" in str(object_name).lower() or "bottle" in str(object_name).lower()):
        print("Tall/narrow object direct pick failed; trying audited side-grasp fallback once.")
        grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Tall side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and candidate_index >= 11:
        print("Direct verified pick exhausted canonical offsets; trying affordance fallback once.")
        grasp = approach_and_grasp_object(object_name, max_attempts=1, start_candidate_index=max(0, candidate_index - 11))
        print("Affordance fallback result:", grasp)
    resolved_name = grasp.get("object_name", object_name) if isinstance(grasp, dict) else object_name
    after = measure_object_grasp_state(resolved_name)
    print("After grasp:", after)
    held = bool(isinstance(grasp, dict) and grasp.get("held")) or bool(isinstance(after, dict) and after.get("held"))
    lift_delta = 0.0
    if isinstance(grasp, dict):
        lift_delta = max(lift_delta, float(grasp.get("lift_delta", 0.0) or 0.0))
    if isinstance(after, dict):
        lift_delta = max(lift_delta, float(after.get("lift_delta", 0.0) or 0.0))
    stable_held = bool(held or lift_delta >= 0.035)
    strict_hold = None
    if fragile_support_grasp:
        strict_hold = measure_object_grasp_state(resolved_name, strict=True)
        stable_held = bool(held and (strict_hold.get("held", False) or lift_delta >= 0.060))
    if drawer_bowl_pick:
        transport_probe = grasp.get("transport_probe") if isinstance(grasp, dict) else None
        if not isinstance(transport_probe, dict) and isinstance(grasp, dict) and isinstance(grasp.get("best_attempt"), dict):
            transport_probe = grasp.get("best_attempt", {}).get("transport_probe")
        probe_held = bool(isinstance(transport_probe, dict) and transport_probe.get("held", False))
        stable_held = bool(held and (probe_held or lift_delta >= 0.070))
    status = "satisfied" if stable_held else "needs_repair"
    evidence = f"held={held}, stable_held={stable_held}, lift_delta={lift_delta:.4f}, object={resolved_name}, target={target_name}, candidate_index={candidate_index}, drawer_level={drawer_level}"
    repair = "" if status == "satisfied" else "Audited single grasp attempt failed; retry stage with next snapshot/candidate."
except Exception as exc:
    status = "needs_repair"
    evidence = f"audited pick-lift fast-path exception: {exc}"
    repair = "Retry stage if episode is still active; otherwise reset trial or reduce motion budget."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 4
STAGE_ID = "obj1_moka_pot_pick_lift_verify"
STAGE_ATTEMPT = 3
import json

stage_id = 'obj1_moka_pot_pick_lift_verify'
object_name = 'moka_pot'
target_name = 'moka_pot'
predicate = 'on'
candidate_index = 2
drawer_bowl_pick = False
drawer_level = None
fragile_support_grasp = bool(
    predicate == "on"
    and (
        any(token in str(object_name).lower() for token in ("mug", "cup", "bowl", "ramekin"))
        or any(token in str(target_name).lower() for token in ("plate", "bowl"))
    )
)

print(f"Stage: {stage_id} - deterministic audited pick-lift for {object_name} candidate_index={candidate_index}")
try:
    drawer_preopen = None
    if drawer_bowl_pick:
        drawer_required_open_fraction = 0.96 if str(drawer_level or "").lower() == "bottom" else 0.98
        drawer_before_guard = get_cabinet_drawer_progress(drawer_level or "middle")
        drawer_before_fraction = float(drawer_before_guard.get("open_fraction", 0.0) if isinstance(drawer_before_guard, dict) else drawer_before_guard)
        drawer_before_bddl_open = bool(isinstance(drawer_before_guard, dict) and drawer_before_guard.get("bddl_open", False))
        if drawer_before_fraction >= drawer_required_open_fraction:
            drawer_preopen = {"reason": "preserve_strictly_open_drawer_for_bowl_insertion", "after": drawer_before_guard, "opened": True}
        else:
            drawer_preopen = open_drawer_by_sleeve_retry(
                drawer_level or "middle",
                open_threshold=drawer_required_open_fraction,
                max_cycles=5,
                test_distance=0.055,
                pull_distance=0.48,
                strict_open_fraction=True,
            )
        print("Drawer pre-pick open guard:", {"drawer_level": drawer_level, "before": drawer_before_guard, "result": drawer_preopen})
        drawer_after_guard = drawer_preopen.get("after", {}) if isinstance(drawer_preopen, dict) else {}
        drawer_after_fraction = float(drawer_after_guard.get("open_fraction", 0.0) or 0.0)
        drawer_after_bddl_open = bool(isinstance(drawer_after_guard, dict) and drawer_after_guard.get("bddl_open", False))
        if drawer_after_fraction < drawer_required_open_fraction:
            evidence = (
                f"drawer pre-pick open guard failed, level={drawer_level}, "
                f"open_fraction={drawer_after_fraction:.3f}, required={drawer_required_open_fraction:.3f}, "
                f"bddl_open={drawer_after_bddl_open}, result={drawer_preopen}"
            )
            raise Exception(evidence)
    before = measure_object_grasp_state(object_name)
    print("Before grasp:", before)
    object_text = str(object_name).lower()
    if drawer_bowl_pick:
        grasp = verified_low_side_pick_object(
            object_name,
            max_attempts=2,
            start_candidate_index=candidate_index,
            lift_height=0.075,
            settle_steps=18,
        )
        print("Drawer bowl low-side pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False):
            print("Drawer bowl low-side pick failed; falling back to geometry pick candidate.")
            grasp = verified_geometry_pick_object(
                object_name,
                max_attempts=1,
                start_candidate_index=max(0, candidate_index - 2),
                lift_height=0.22,
            )
            print("Drawer bowl geometry fallback result:", grasp)
    elif "moka" in object_text or "book" in object_text:
        grasp = verified_geometry_pick_object(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Geometry pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False) and "moka" in object_text:
            print("Moka geometry pick was not transport-stable; trying affordance grasp fallback.")
            grasp = approach_and_grasp_object(
                object_name,
                max_attempts=2,
                start_candidate_index=max(0, candidate_index % 6),
                lift_height=0.135,
            )
            print("Moka affordance fallback result:", grasp)
    else:
        direct_attempts = 3 if drawer_bowl_pick else 1
        grasp = verified_lift_pick_object(object_name, max_attempts=direct_attempts, start_candidate_index=candidate_index)
        print("Audited grasp result:", grasp)
    if isinstance(grasp, dict) and (not drawer_bowl_pick) and (("bowl" in str(object_name).lower()) or fragile_support_grasp):
        strict_after_direct = measure_object_grasp_state(grasp.get("object_name", object_name), strict=True)
        direct_lift = float(grasp.get("lift_delta", 0.0) or 0.0)
        direct_stable = bool(grasp.get("held", False) and (strict_after_direct.get("held", False) or direct_lift >= 0.060))
        if not direct_stable:
            print("Direct pick is not transport-stable for fragile support placement; trying affordance fallback.")
            fallback_index = max(0, candidate_index)
            grasp = approach_and_grasp_object(object_name, max_attempts=2, start_candidate_index=fallback_index)
            print("Fragile support affordance fallback result:", grasp)
            if isinstance(grasp, dict) and not grasp.get("held", False) and ("mug" in str(object_name).lower()):
                print("Mug affordance fallback failed; trying side-grasp fallback.")
                grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
                print("Mug side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and ("wine" in str(object_name).lower() or "bottle" in str(object_name).lower()):
        print("Tall/narrow object direct pick failed; trying audited side-grasp fallback once.")
        grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Tall side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and candidate_index >= 11:
        print("Direct verified pick exhausted canonical offsets; trying affordance fallback once.")
        grasp = approach_and_grasp_object(object_name, max_attempts=1, start_candidate_index=max(0, candidate_index - 11))
        print("Affordance fallback result:", grasp)
    resolved_name = grasp.get("object_name", object_name) if isinstance(grasp, dict) else object_name
    after = measure_object_grasp_state(resolved_name)
    print("After grasp:", after)
    held = bool(isinstance(grasp, dict) and grasp.get("held")) or bool(isinstance(after, dict) and after.get("held"))
    lift_delta = 0.0
    if isinstance(grasp, dict):
        lift_delta = max(lift_delta, float(grasp.get("lift_delta", 0.0) or 0.0))
    if isinstance(after, dict):
        lift_delta = max(lift_delta, float(after.get("lift_delta", 0.0) or 0.0))
    stable_held = bool(held or lift_delta >= 0.035)
    strict_hold = None
    if fragile_support_grasp:
        strict_hold = measure_object_grasp_state(resolved_name, strict=True)
        stable_held = bool(held and (strict_hold.get("held", False) or lift_delta >= 0.060))
    if drawer_bowl_pick:
        transport_probe = grasp.get("transport_probe") if isinstance(grasp, dict) else None
        if not isinstance(transport_probe, dict) and isinstance(grasp, dict) and isinstance(grasp.get("best_attempt"), dict):
            transport_probe = grasp.get("best_attempt", {}).get("transport_probe")
        probe_held = bool(isinstance(transport_probe, dict) and transport_probe.get("held", False))
        stable_held = bool(held and (probe_held or lift_delta >= 0.070))
    status = "satisfied" if stable_held else "needs_repair"
    evidence = f"held={held}, stable_held={stable_held}, lift_delta={lift_delta:.4f}, object={resolved_name}, target={target_name}, candidate_index={candidate_index}, drawer_level={drawer_level}"
    repair = "" if status == "satisfied" else "Audited single grasp attempt failed; retry stage with next snapshot/candidate."
except Exception as exc:
    status = "needs_repair"
    evidence = f"audited pick-lift fast-path exception: {exc}"
    repair = "Retry stage if episode is still active; otherwise reset trial or reduce motion budget."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 5
STAGE_ID = "obj1_moka_pot_pick_lift_verify"
STAGE_ATTEMPT = 4
import json

stage_id = 'obj1_moka_pot_pick_lift_verify'
object_name = 'moka_pot'
target_name = 'moka_pot'
predicate = 'on'
candidate_index = 3
drawer_bowl_pick = False
drawer_level = None
fragile_support_grasp = bool(
    predicate == "on"
    and (
        any(token in str(object_name).lower() for token in ("mug", "cup", "bowl", "ramekin"))
        or any(token in str(target_name).lower() for token in ("plate", "bowl"))
    )
)

print(f"Stage: {stage_id} - deterministic audited pick-lift for {object_name} candidate_index={candidate_index}")
try:
    drawer_preopen = None
    if drawer_bowl_pick:
        drawer_required_open_fraction = 0.96 if str(drawer_level or "").lower() == "bottom" else 0.98
        drawer_before_guard = get_cabinet_drawer_progress(drawer_level or "middle")
        drawer_before_fraction = float(drawer_before_guard.get("open_fraction", 0.0) if isinstance(drawer_before_guard, dict) else drawer_before_guard)
        drawer_before_bddl_open = bool(isinstance(drawer_before_guard, dict) and drawer_before_guard.get("bddl_open", False))
        if drawer_before_fraction >= drawer_required_open_fraction:
            drawer_preopen = {"reason": "preserve_strictly_open_drawer_for_bowl_insertion", "after": drawer_before_guard, "opened": True}
        else:
            drawer_preopen = open_drawer_by_sleeve_retry(
                drawer_level or "middle",
                open_threshold=drawer_required_open_fraction,
                max_cycles=5,
                test_distance=0.055,
                pull_distance=0.48,
                strict_open_fraction=True,
            )
        print("Drawer pre-pick open guard:", {"drawer_level": drawer_level, "before": drawer_before_guard, "result": drawer_preopen})
        drawer_after_guard = drawer_preopen.get("after", {}) if isinstance(drawer_preopen, dict) else {}
        drawer_after_fraction = float(drawer_after_guard.get("open_fraction", 0.0) or 0.0)
        drawer_after_bddl_open = bool(isinstance(drawer_after_guard, dict) and drawer_after_guard.get("bddl_open", False))
        if drawer_after_fraction < drawer_required_open_fraction:
            evidence = (
                f"drawer pre-pick open guard failed, level={drawer_level}, "
                f"open_fraction={drawer_after_fraction:.3f}, required={drawer_required_open_fraction:.3f}, "
                f"bddl_open={drawer_after_bddl_open}, result={drawer_preopen}"
            )
            raise Exception(evidence)
    before = measure_object_grasp_state(object_name)
    print("Before grasp:", before)
    object_text = str(object_name).lower()
    if drawer_bowl_pick:
        grasp = verified_low_side_pick_object(
            object_name,
            max_attempts=2,
            start_candidate_index=candidate_index,
            lift_height=0.075,
            settle_steps=18,
        )
        print("Drawer bowl low-side pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False):
            print("Drawer bowl low-side pick failed; falling back to geometry pick candidate.")
            grasp = verified_geometry_pick_object(
                object_name,
                max_attempts=1,
                start_candidate_index=max(0, candidate_index - 2),
                lift_height=0.22,
            )
            print("Drawer bowl geometry fallback result:", grasp)
    elif "moka" in object_text or "book" in object_text:
        grasp = verified_geometry_pick_object(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Geometry pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False) and "moka" in object_text:
            print("Moka geometry pick was not transport-stable; trying affordance grasp fallback.")
            grasp = approach_and_grasp_object(
                object_name,
                max_attempts=2,
                start_candidate_index=max(0, candidate_index % 6),
                lift_height=0.135,
            )
            print("Moka affordance fallback result:", grasp)
    else:
        direct_attempts = 3 if drawer_bowl_pick else 1
        grasp = verified_lift_pick_object(object_name, max_attempts=direct_attempts, start_candidate_index=candidate_index)
        print("Audited grasp result:", grasp)
    if isinstance(grasp, dict) and (not drawer_bowl_pick) and (("bowl" in str(object_name).lower()) or fragile_support_grasp):
        strict_after_direct = measure_object_grasp_state(grasp.get("object_name", object_name), strict=True)
        direct_lift = float(grasp.get("lift_delta", 0.0) or 0.0)
        direct_stable = bool(grasp.get("held", False) and (strict_after_direct.get("held", False) or direct_lift >= 0.060))
        if not direct_stable:
            print("Direct pick is not transport-stable for fragile support placement; trying affordance fallback.")
            fallback_index = max(0, candidate_index)
            grasp = approach_and_grasp_object(object_name, max_attempts=2, start_candidate_index=fallback_index)
            print("Fragile support affordance fallback result:", grasp)
            if isinstance(grasp, dict) and not grasp.get("held", False) and ("mug" in str(object_name).lower()):
                print("Mug affordance fallback failed; trying side-grasp fallback.")
                grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
                print("Mug side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and ("wine" in str(object_name).lower() or "bottle" in str(object_name).lower()):
        print("Tall/narrow object direct pick failed; trying audited side-grasp fallback once.")
        grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Tall side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and candidate_index >= 11:
        print("Direct verified pick exhausted canonical offsets; trying affordance fallback once.")
        grasp = approach_and_grasp_object(object_name, max_attempts=1, start_candidate_index=max(0, candidate_index - 11))
        print("Affordance fallback result:", grasp)
    resolved_name = grasp.get("object_name", object_name) if isinstance(grasp, dict) else object_name
    after = measure_object_grasp_state(resolved_name)
    print("After grasp:", after)
    held = bool(isinstance(grasp, dict) and grasp.get("held")) or bool(isinstance(after, dict) and after.get("held"))
    lift_delta = 0.0
    if isinstance(grasp, dict):
        lift_delta = max(lift_delta, float(grasp.get("lift_delta", 0.0) or 0.0))
    if isinstance(after, dict):
        lift_delta = max(lift_delta, float(after.get("lift_delta", 0.0) or 0.0))
    stable_held = bool(held or lift_delta >= 0.035)
    strict_hold = None
    if fragile_support_grasp:
        strict_hold = measure_object_grasp_state(resolved_name, strict=True)
        stable_held = bool(held and (strict_hold.get("held", False) or lift_delta >= 0.060))
    if drawer_bowl_pick:
        transport_probe = grasp.get("transport_probe") if isinstance(grasp, dict) else None
        if not isinstance(transport_probe, dict) and isinstance(grasp, dict) and isinstance(grasp.get("best_attempt"), dict):
            transport_probe = grasp.get("best_attempt", {}).get("transport_probe")
        probe_held = bool(isinstance(transport_probe, dict) and transport_probe.get("held", False))
        stable_held = bool(held and (probe_held or lift_delta >= 0.070))
    status = "satisfied" if stable_held else "needs_repair"
    evidence = f"held={held}, stable_held={stable_held}, lift_delta={lift_delta:.4f}, object={resolved_name}, target={target_name}, candidate_index={candidate_index}, drawer_level={drawer_level}"
    repair = "" if status == "satisfied" else "Audited single grasp attempt failed; retry stage with next snapshot/candidate."
except Exception as exc:
    status = "needs_repair"
    evidence = f"audited pick-lift fast-path exception: {exc}"
    repair = "Retry stage if episode is still active; otherwise reset trial or reduce motion budget."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 6
STAGE_ID = "obj1_moka_pot_transport_guarded"
STAGE_ATTEMPT = 1
import json

stage_id = 'obj1_moka_pot_transport_guarded'
object_name = 'moka_pot'
target_name = 'flat_stove_1_cook_region'
predicate = 'on'

print(f"Stage: {stage_id} - deterministic guarded transport for {object_name} to {target_name}")
try:
    transport = guarded_transport_held_object_to_target(
        object_name,
        predicate=predicate,
        target_name=target_name or None,
    )
    print("Guarded transport result:", transport)
    effective_predicate = str(transport.get("target", {}).get("predicate", predicate)) if isinstance(transport, dict) else predicate
    held_before = bool(isinstance(transport, dict) and transport.get("held_before"))
    held_after = bool(isinstance(transport, dict) and transport.get("held_after"))
    arrived = bool(isinstance(transport, dict) and transport.get("arrived"))
    fused_satisfied = bool(isinstance(transport, dict) and transport.get("relation_satisfied"))
    task_status = get_task_status()
    if isinstance(transport, dict) and isinstance(transport.get("task_status"), dict):
        task_status = transport.get("task_status", task_status)
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    xy_distance = float(transport.get("xy_distance_to_target", 999.0) if isinstance(transport, dict) else 999.0)
    if isinstance(transport, dict) and xy_distance >= 999.0:
        relation_for_xy = transport.get("final_relation") or transport.get("relation") or {}
        if isinstance(relation_for_xy, dict) and "local_delta" in relation_for_xy:
            try:
                local_delta_for_xy = relation_for_xy.get("local_delta")
                xy_distance = float((float(local_delta_for_xy[0]) ** 2 + float(local_delta_for_xy[1]) ** 2) ** 0.5)
            except Exception:
                pass
    closure = None
    closure_xy_threshold = 0.045 if effective_predicate == "on" else 0.090
    tolerance = float(transport.get("xy_tolerance", 0.075) if isinstance(transport, dict) else 0.075)
    held_near_release = bool(held_after and effective_predicate in {"on", "in"} and xy_distance <= max(0.090, 1.8 * tolerance))
    strict_caddy_transport = bool(
        effective_predicate == "in"
        and "book" in str(object_name).lower()
        and ("caddy" in str(target_name).lower() or "compartment" in str(target_name).lower())
    )
    final_relation = transport.get("final_relation", None) if isinstance(transport, dict) else None
    z_only_near_success = bool(isinstance(transport, dict) and transport.get("z_only_near_success", False))
    near_contact_relation = bool(isinstance(transport, dict) and transport.get("near_contact_relation"))
    drawer_immediate_release_failed = False
    if (not near_contact_relation) and isinstance(final_relation, dict) and effective_predicate == "in":
        relation_xy = final_relation.get("xy_distance", None)
        if relation_xy is None and "local_delta" in final_relation:
            try:
                local_delta = final_relation.get("local_delta")
                relation_xy = float((float(local_delta[0]) ** 2 + float(local_delta[1]) ** 2) ** 0.5)
            except Exception:
                relation_xy = None
        near_contact_relation = bool(
            final_relation.get("contact", False)
            and "local_delta" in final_relation
            and relation_xy is not None
            and float(relation_xy) <= 0.18
        )
    near_target_for_closure = (
        (not held_after)
        and effective_predicate in {"on", "in"}
        and (xy_distance <= closure_xy_threshold or near_contact_relation)
        and not completed
        and reward < 1.0
        and (not drawer_immediate_release_failed or z_only_near_success or near_contact_relation)
    )
    if (
        predicate == "in"
        and ("drawer" in str(target_name).lower() or "cabinet" in str(target_name).lower())
        and not held_before
    ):
        near_target_for_closure = False
    if near_target_for_closure:
        print("Transport lost hold near target; routing to release/local-closure stage.")
    if held_near_release and not arrived:
        print("Transport still holds object near target; allowing release/local-closure stage.")
    closure_satisfied = bool(isinstance(closure, dict) and closure.get("relation_satisfied"))
    closure_attempted_near_target = bool(isinstance(closure, dict) and closure.get("near_target"))
    def summarize_relation(relation):
        if not isinstance(relation, dict):
            return None
        def as_list(value, limit=3):
            try:
                return [float(value[i]) for i in range(min(limit, len(value)))]
            except Exception:
                try:
                    seq = list(value)
                    return [float(x) for x in seq[:limit]]
                except Exception:
                    return None
        return {
            "satisfied": bool(relation.get("satisfied", relation.get("in", relation.get("on", False)))),
            "in": relation.get("in"),
            "on": relation.get("on"),
            "contact": relation.get("contact"),
            "local_delta": as_list(relation.get("local_delta")),
            "site_local_delta": as_list(relation.get("site_local_delta")),
            "site_size_abs": as_list(relation.get("site_size_abs")),
            "aabb_margin_lower_world": as_list(relation.get("aabb_margin_lower_world")),
            "aabb_margin_upper_world": as_list(relation.get("aabb_margin_upper_world")),
            "aabb_z_excess": relation.get("aabb_z_excess"),
            "aabb_z_deficit": relation.get("aabb_z_deficit"),
            "aabb_margin_min": relation.get("aabb_margin_min"),
            "object_pos_world": as_list(relation.get("object_pos_world")),
            "site_pos_world": as_list(relation.get("site_pos_world")),
        }
    def summarize_nested_record(record):
        if not isinstance(record, dict):
            return {}
        return {
            "action": record.get("action"),
            "iteration": record.get("iteration"),
            "trigger": record.get("trigger"),
            "before_norm": record.get("before_norm"),
            "after_norm": record.get("after_norm"),
            "object_motion_xy": record.get("object_motion_xy"),
            "stroke": record.get("stroke"),
            "push_z": record.get("push_z"),
            "improved": record.get("improved"),
            "worsened": record.get("worsened"),
            "local_delta_before": summarize_relation(record.get("relation_before")).get("local_delta") if isinstance(summarize_relation(record.get("relation_before")), dict) else None,
            "relation_after": summarize_relation(record.get("relation_after") or record.get("relation")),
            "reason": record.get("reason"),
            "error": record.get("error"),
        }
    def summarize_move(move):
        if not isinstance(move, dict):
            return None
        return {
            "success": move.get("success"),
            "distance": move.get("distance"),
            "error": move.get("error"),
            "reason": move.get("reason"),
        }
    def summarize_state(state):
        if not isinstance(state, dict):
            return None
        def as_list(value, limit=3):
            try:
                return [float(value[i]) for i in range(min(limit, len(value)))]
            except Exception:
                try:
                    seq = list(value)
                    return [float(x) for x in seq[:limit]]
                except Exception:
                    return None
        return {
            "held": state.get("held"),
            "object_position": as_list(state.get("object_position")),
            "finger_midpoint": as_list(state.get("finger_midpoint")),
        }
    def summarize_waypoint_record(record):
        if not isinstance(record, dict):
            return {}
        def as_list(value, limit=3):
            try:
                return [float(value[i]) for i in range(min(limit, len(value)))]
            except Exception:
                try:
                    seq = list(value)
                    return [float(x) for x in seq[:limit]]
                except Exception:
                    return None
        align = record.get("align")
        align_summary = None
        if isinstance(align, dict):
            align_records = align.get("records", [])
            align_tail = []
            if isinstance(align_records, list):
                for align_record in align_records[-2:]:
                    if not isinstance(align_record, dict):
                        continue
                    align_tail.append({
                        "iteration": align_record.get("iteration"),
                        "action": align_record.get("action"),
                        "target_object_position": as_list(align_record.get("target_object_position")),
                        "target_finger_midpoint": as_list(align_record.get("target_finger_midpoint")),
                        "object_position_after": as_list(align_record.get("object_position_after")),
                        "error_after": as_list(align_record.get("error_after")),
                        "held_after": align_record.get("held_after"),
                    })
            align_summary = {
                "arrived": align.get("arrived"),
                "held_before": align.get("held_before"),
                "held_after": align.get("held_after"),
                "xy_error": align.get("xy_error"),
                "z_error": align.get("z_error"),
                "target_object_position": as_list(align.get("target_object_position")),
                "tail": align_tail,
            }
        return {
            "phase": record.get("phase"),
            "reason": record.get("reason"),
            "skipped": record.get("skipped"),
            "target_object_position": as_list(record.get("target_object_position")),
            "target_ee_position": as_list(record.get("target_ee_position")),
            "object_to_ee": as_list(record.get("object_to_ee")),
            "object_error_after": as_list(record.get("object_error_after")),
            "move": summarize_move(record.get("move")),
            "align": align_summary,
            "state_after": summarize_state(record.get("state_after") or record.get("state")),
            "relation_after": summarize_relation(record.get("relation_after") or record.get("relation")),
        }
    def summarize_align_item(item):
        if not isinstance(item, dict):
            return {}
        def as_list(value, limit=3):
            try:
                return [float(value[i]) for i in range(min(limit, len(value)))]
            except Exception:
                try:
                    seq = list(value)
                    return [float(x) for x in seq[:limit]]
                except Exception:
                    return None
        record = item.get("record", {})
        record_summary = {}
        if isinstance(record, dict):
            record_summary = {
                "skill": record.get("skill"),
                "reason": record.get("reason"),
                "relation_satisfied": record.get("relation_satisfied"),
                "held_before": record.get("held_before"),
                "held_after": record.get("held_after"),
                "released": record.get("released"),
                "arrived": record.get("arrived"),
                "xy_error": record.get("xy_error"),
                "z_error": record.get("z_error"),
                "target_position": as_list(record.get("target_position")),
                "release_target": as_list(record.get("release_target")),
                "pull_direction": as_list(record.get("pull_direction")),
                "relation": summarize_relation(record.get("relation")),
                "final_relation": summarize_relation(record.get("final_relation")),
                "task_status": record.get("task_status"),
                "after": summarize_state(record.get("after")),
            }
            nested = record.get("records", [])
            if isinstance(nested, list):
                record_summary["record_tail"] = [
                    summarize_waypoint_record(rec)
                    for rec in nested[-4:]
                    if isinstance(rec, dict)
                ]
        candidate_records = item.get("records", [])
        candidate_tail = []
        if isinstance(candidate_records, list):
            for candidate in candidate_records[-3:]:
                if not isinstance(candidate, dict):
                    continue
                candidate_tail.append({
                    "candidate_index": candidate.get("candidate_index"),
                    "depth": candidate.get("depth"),
                    "lateral": candidate.get("lateral"),
                    "skipped": candidate.get("skipped"),
                    "reason": candidate.get("reason"),
                    "object_error_to_target": as_list(candidate.get("object_error_to_target")),
                    "held": (candidate.get("record") or {}).get("held") if isinstance(candidate.get("record"), dict) else None,
                    "relation_before_gripper_release": summarize_relation(candidate.get("relation_before_gripper_release")),
                })
        return {
            "phase": item.get("phase"),
            "release_strategy": item.get("release_strategy"),
            "held_after": item.get("held_after"),
            "drawer_level": item.get("drawer_level"),
            "require_drawer_closing_clearance": item.get("require_drawer_closing_clearance"),
            "relation": summarize_relation(item.get("relation")),
            "record": record_summary,
            "candidate_tail": candidate_tail,
        }
    record_actions = []
    if isinstance(transport, dict):
        for item in transport.get("records", [])[-6:]:
            if isinstance(item, dict):
                result = item.get("result", {})
                result_summary = {}
                if isinstance(result, dict):
                    result_summary = {
                        "reason": result.get("reason"),
                        "relation_satisfied": result.get("relation_satisfied"),
                        "satisfied": result.get("satisfied"),
                        "near_contact_relation": result.get("near_contact_relation"),
                        "skill": result.get("skill"),
                        "xy_distance_to_target": result.get("xy_distance_to_target"),
                        "xy_tolerance": result.get("xy_tolerance"),
                        "final_relation": summarize_relation(result.get("final_relation")),
                    }
                    nested_records = result.get("records", [])
                    if isinstance(nested_records, list):
                        result_summary["record_tail"] = [
                            summarize_nested_record(rec)
                            for rec in nested_records[-4:]
                            if isinstance(rec, dict)
                        ]
                    align_records = result.get("align_records", [])
                    if isinstance(align_records, list):
                        result_summary["align_tail"] = [
                            summarize_align_item(rec)
                            for rec in align_records[-6:]
                            if isinstance(rec, dict)
                        ]
                record_actions.append({"action": item.get("action"), "result": result_summary})
    if (
        predicate == "in"
        and ("drawer" in str(target_name).lower() or "cabinet" in str(target_name).lower())
        and not held_before
    ):
        status = "needs_repair"
    else:
        status = "satisfied" if (
            fused_satisfied
            or z_only_near_success
            or near_target_for_closure
            or (held_after and (arrived or (held_near_release and not strict_caddy_transport)))
            or completed
            or reward >= 1.0
        ) else "needs_repair"
    evidence = (
        f"fused_satisfied={fused_satisfied}, z_only_near_success={z_only_near_success}, drawer_immediate_release_failed={drawer_immediate_release_failed}, held_before={held_before}, held_after={held_after}, arrived={arrived}, held_near_release={held_near_release}, near_contact_relation={near_contact_relation}, near_target_closure={near_target_for_closure}, "
        f"closure_satisfied={closure_satisfied}, reward={reward:.3f}, task_completed={completed}, "
        f"xy_distance={xy_distance:.4f}, xy_tolerance={tolerance:.4f}, final_relation={final_relation}, "
        f"record_actions={record_actions}, object={object_name}, target={target_name}, predicate={effective_predicate}"
    )
    if status == "satisfied":
        repair = ""
    elif (
        predicate == "in"
        and ("drawer" in str(target_name).lower() or "cabinet" in str(target_name).lower())
        and not held_before
    ):
        repair = "Object is not verified-held before drawer transport; replan pick_lift_verify before any placement repair."
    elif strict_caddy_transport and (not held_after or not arrived):
        repair = "Object is not verified-held/aligned for caddy transport; return control to the pick_lift stage and reacquire before release."
    else:
        repair = "Guarded transport did not reach a verified pre-release pose; keep holding the object and retry transport before release."
except Exception as exc:
    status = "needs_repair"
    evidence = f"guarded transport fast-path exception: {exc}"
    repair = "Retry transport after checking target grounding and grasp state."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 7
STAGE_ID = "obj1_moka_pot_low_release_verify"
STAGE_ATTEMPT = 1
import json

stage_id = 'obj1_moka_pot_low_release_verify'
object_name = 'moka_pot'
target_name = 'flat_stove_1_cook_region'
predicate = 'on'
is_final_stage = True
require_drawer_closing_clearance = False

print(f"Stage: {stage_id} - deterministic low release for {object_name} to {target_name} predicate={predicate}")
try:
    used_microwave_release = bool(
        predicate == "in"
        and "microwave" in str(target_name).lower()
    )
    used_rack_release = bool(
        predicate in {"on", "in_or_on"}
        and ("rack" in str(target_name).lower() or "caddy" in str(target_name).lower())
    )
    used_thin_caddy_release = bool(
        predicate == "in"
        and ("caddy" in str(target_name).lower() or "compartment" in str(target_name).lower())
    )
    used_drawer_release = bool(
        predicate == "in"
        and ("drawer" in str(target_name).lower() or "cabinet" in str(target_name).lower())
    )
    drawer_precondition_failure = None
    if used_drawer_release:
        try:
            target_text_for_drawer = str(target_name).lower()
            if "bottom" in target_text_for_drawer:
                drawer_level = "bottom"
            elif "top" in target_text_for_drawer:
                drawer_level = "top"
            else:
                drawer_level = "middle"
            drawer_progress = get_cabinet_drawer_progress(drawer_level)
            drawer_open_fraction = float(drawer_progress.get("open_fraction", 0.0) if isinstance(drawer_progress, dict) else drawer_progress)
            drawer_is_closed = bool(isinstance(drawer_progress, dict) and drawer_progress.get("bddl_close", False))
            if drawer_is_closed or drawer_open_fraction < 0.72:
                drawer_precondition_failure = {
                    "level": drawer_level,
                    "progress": drawer_progress,
                    "open_fraction": drawer_open_fraction,
                    "is_closed": drawer_is_closed,
                    "reason": "drawer is not physically open enough for insertion",
                }
        except Exception as drawer_pre_exc:
            drawer_precondition_failure = {"error": repr(drawer_pre_exc)}
    drawer_prefilter = None
    if used_drawer_release and drawer_precondition_failure is None:
        try:
            goal_diag_before_release = get_bddl_goal_diagnostics()
            goals_before_release = goal_diag_before_release.get("goal_state", []) if isinstance(goal_diag_before_release, dict) else []
            for goal in goals_before_release:
                if not isinstance(goal, dict):
                    continue
                raw = goal.get("raw", [])
                if len(raw) != 3 or str(raw[0]).lower() != "in":
                    continue
                if object_name not in str(raw[1]).lower() and str(raw[1]).lower() not in object_name:
                    continue
                diag = goal.get("diagnostic", {})
                if not isinstance(diag, dict):
                    continue
                axes = diag.get("aabb_inside_axes", [])
                xy_inside = bool(len(axes) >= 2 and bool(axes[0]) and bool(axes[1]))
                z_excess = float(diag.get("aabb_z_excess", 0.0) or 0.0)
                z_deficit = float(diag.get("aabb_z_deficit", 0.0) or 0.0)
                contact = bool(diag.get("contact", False))
                if xy_inside and contact and z_deficit <= 0.002 and 0.0 < z_excess <= 0.020:
                    drawer_prefilter = {
                        "target_name": str(goal.get("target_name") or raw[2]),
                        "diagnostic": diag,
                        "z_excess": z_excess,
                    }
                    break
        except Exception as prefilter_exc:
            drawer_prefilter = {"error": repr(prefilter_exc)}
    if isinstance(drawer_precondition_failure, dict):
        print("Drawer insertion precondition failed before low release:", drawer_precondition_failure)
        release = {
            "object_name": object_name,
            "target": {"predicate": predicate, "target_name": target_name},
            "before": {},
            "after": {},
            "relation": {},
            "task_status": get_task_status(),
            "relation_satisfied": False,
            "needs_repair": True,
            "drawer_precondition_failure": drawer_precondition_failure,
            "reason": "drawer insertion precondition failed; reopen drawer before placing object",
            "skill": "low_release_object_to_target_precondition",
        }
    elif isinstance(drawer_prefilter, dict) and drawer_prefilter.get("diagnostic") is not None:
        print("Drawer z-only near-success detected before low release; running bounded z-only settle.", drawer_prefilter)
        release = drawer_z_only_settle_object(
            object_name,
            target_name=drawer_prefilter.get("target_name") or target_name or None,
            max_attempts=2,
            max_z_excess=0.020,
            settle_steps=26,
        )
    elif used_microwave_release:
        release = place_object_in_microwave_region(
            object_name,
            target_name=target_name or "microwave",
        )
    elif used_rack_release:
        release = rack_supported_release_object_to_target(
            object_name,
            target_name=target_name or None,
        )
    elif used_thin_caddy_release:
        caddy_hold = measure_object_grasp_state(object_name)
        if not bool(isinstance(caddy_hold, dict) and caddy_hold.get("held", False)):
            release = low_release_object_to_target(
                object_name,
                predicate=predicate,
                target_name=target_name or None,
                settle_steps=12,
            )
            if isinstance(release, dict):
                release["before"] = caddy_hold
                release["needs_repick"] = True
                release["reason"] = (
                    "object is not verified-held before caddy release; "
                    "return control to the pick_lift stage for reacquire-transport-release"
                )
        else:
            release = low_release_object_to_target(
                object_name,
                predicate=predicate,
                target_name=target_name or None,
                settle_steps=40,
            )
    else:
        release = low_release_object_to_target(
            object_name,
            predicate=predicate,
            target_name=target_name or None,
            require_drawer_closing_clearance=bool(require_drawer_closing_clearance),
        )
    if used_drawer_release and isinstance(drawer_precondition_failure, dict):
        print("Drawer release precondition result:", release)
    elif used_microwave_release and isinstance(release, dict):
        print("Microwave release result:", release)
    elif used_rack_release and isinstance(release, dict):
        print("Rack release summary:", release.get("summary", release))
    elif used_thin_caddy_release and isinstance(release, dict):
        print("Thin caddy low release result:", release)
    elif isinstance(drawer_prefilter, dict) and drawer_prefilter.get("diagnostic") is not None:
        print("Drawer z-only settle result:", release)
    else:
        print("Low release result:", release)
    effective_predicate = str(release.get("target", {}).get("predicate", predicate)) if isinstance(release, dict) else predicate
    preserved_drawer_z_only = bool(
        isinstance(release, dict)
        and release.get("z_only_near_success_preserved", False)
    )
    satisfied = bool(isinstance(release, dict) and release.get("relation_satisfied"))
    microwave_near_inside_z_only = False
    if used_microwave_release and isinstance(release, dict) and not satisfied:
        final_relation_for_microwave = release.get("final_relation") or release.get("relation") or {}
        if isinstance(final_relation_for_microwave, dict):
            try:
                local_delta = final_relation_for_microwave.get("local_delta")
                site_size = final_relation_for_microwave.get(
                    "site_size_abs",
                    final_relation_for_microwave.get("site_size"),
                )
                z_excess = float(final_relation_for_microwave.get("aabb_z_excess", 0.0) or 0.0)
                microwave_near_inside_z_only = bool(
                    final_relation_for_microwave.get("contact", False)
                    and len(local_delta) >= 3
                    and len(site_size) >= 3
                    and abs(float(local_delta[0])) <= max(abs(float(site_size[0])) + 0.018, 0.060)
                    and abs(float(local_delta[1])) <= max(abs(float(site_size[1])) + 0.018, 0.060)
                    and 0.0 < z_excess <= 0.025
                )
            except Exception:
                microwave_near_inside_z_only = False
    drawer_clearance = release.get("drawer_closing_clearance", None) if isinstance(release, dict) else None
    if (
        satisfied
        and bool(require_drawer_closing_clearance)
        and effective_predicate == "in"
        and isinstance(drawer_clearance, dict)
        and not bool(drawer_clearance.get("has_clearance", False))
    ):
        print("Release satisfies symbolic In/contact; downstream drawer-close clearance is a warning for the close stage:", drawer_clearance)
    repair = None
    if (not satisfied) and used_microwave_release:
        print("Microwave interior release did not satisfy relation; keeping failure local to microwave primitive.")
    elif (not satisfied) and used_rack_release:
        print("Rack/caddy release did not satisfy relation; keeping failure local to rack primitive.")
    elif (not satisfied) and used_thin_caddy_release:
        print("Thin caddy low release did not satisfy relation; entering caddy-specific local closure.")
    drawer_relation_for_repair = {}
    if used_drawer_release and isinstance(release, dict):
        candidate_relation = release.get("final_relation") or release.get("relation") or release.get("after") or {}
        if isinstance(candidate_relation, dict):
            drawer_relation_for_repair = candidate_relation
    drawer_clearance_locally_repairable = bool(
        used_drawer_release
        and bool(require_drawer_closing_clearance)
        and isinstance(drawer_clearance, dict)
        and not bool(drawer_clearance.get("has_clearance", False))
        and isinstance(drawer_relation_for_repair, dict)
        and (
            bool(drawer_relation_for_repair.get("satisfied", False))
            or bool(drawer_relation_for_repair.get("in", False))
            or (
                bool(drawer_relation_for_repair.get("contact", False))
                and "local_delta" in drawer_relation_for_repair
            )
        )
    )
    drawer_min_front_clearance_for_repair = None
    if isinstance(drawer_clearance, dict):
        try:
            drawer_min_front_clearance_for_repair = float(drawer_clearance.get("min_front_clearance"))
        except Exception:
            drawer_min_front_clearance_for_repair = None
    if (
        (not satisfied)
        and used_drawer_release
        and not isinstance(drawer_precondition_failure, dict)
        and drawer_clearance_locally_repairable
    ):
        print("Drawer low release is inside/near the box but lacks closing-clearance margin; preserve placement and let close stage verify.", {
            "relation": drawer_relation_for_repair,
            "drawer_clearance": drawer_clearance,
        })
        repair = {
            "relation_satisfied": True,
            "needs_replan_deeper_held_release": False,
            "reason": "drawer_close_clearance_warning_defer_to_close_stage",
            "initial_relation": drawer_relation_for_repair,
            "final_relation": drawer_relation_for_repair,
            "drawer_closing_clearance": drawer_clearance,
            "near_target": True,
            "xy_distance_to_target": None,
            "task_status": release.get("task_status", get_task_status()) if isinstance(release, dict) else get_task_status(),
        }
        satisfied = True
    elif (
        (not satisfied)
        and used_drawer_release
        and not isinstance(drawer_precondition_failure, dict)
    ):
        relation_for_replan = release.get("relation", {}) if isinstance(release, dict) else {}
        print("Drawer low release failed; preserving state and returning relation feedback instead of unheld local closure.", relation_for_replan)
        repair = {
            "relation_satisfied": False,
            "reason": "drawer_low_release_failed_preserve_state_for_repick",
            "initial_relation": relation_for_replan,
            "final_relation": relation_for_replan,
            "task_status": release.get("task_status", get_task_status()) if isinstance(release, dict) else get_task_status(),
        }
    elif (
        (not satisfied)
        and not used_microwave_release
        and not used_rack_release
        and not (
            used_thin_caddy_release
            and isinstance(release, dict)
            and bool(release.get("needs_repick", False))
        )
        and not isinstance(drawer_precondition_failure, dict)
    ):
        before = release.get("before", {}) if isinstance(release, dict) else {}
        print("Low release not satisfied; entering bounded local placement closure.", before)
        skip_closure_for_preservation = False
        if effective_predicate == "on" and ("stove" in str(target_name).lower() or "cook_region" in str(target_name).lower()):
            try:
                goal_diag_for_closure = get_bddl_goal_diagnostics()
                placement_goals_for_closure = goal_diag_for_closure.get("goal_state", []) if isinstance(goal_diag_for_closure, dict) else []
                skip_closure_for_preservation = bool(
                    any(
                        isinstance(goal, dict)
                        and len(goal.get("raw", [])) == 3
                        and str((goal.get("raw") or [""])[0]).lower() == "on"
                        and bool(goal.get("satisfied", False))
                        for goal in placement_goals_for_closure
                    )
                )
            except Exception:
                skip_closure_for_preservation = False
        if skip_closure_for_preservation:
            print("Skipping local placement closure to preserve already satisfied shared-region placement goals.")
            repair = {"relation_satisfied": False, "reason": "preserve_shared_region_goals"}
            satisfied = False
        else:
            closure_kwargs = {}
            if effective_predicate == "in" and "basket" in str(target_name).lower():
                try:
                    affordance = get_object_grasp_affordance(object_name)
                    extent = affordance.get("geometry", {}).get("extent", [0.0, 0.0, 0.0])
                    max_xy = max(abs(float(extent[0])), abs(float(extent[1])))
                    z_extent = abs(float(extent[2]))
                    large_box_like = bool(max_xy >= 0.150 and z_extent <= max_xy + 0.008)
                except Exception:
                    large_box_like = False
                closure_kwargs = {"near_tolerance": 0.18, "max_iters": 3} if large_box_like else {"max_iters": 2}
            elif used_drawer_release:
                closure_kwargs = {"near_tolerance": 0.32, "max_iters": 2, "settle_steps": 28}
            elif effective_predicate == "in" and ("caddy" in str(target_name).lower() or "compartment" in str(target_name).lower()):
                closure_kwargs = {"near_tolerance": 0.24, "max_iters": 4, "settle_steps": 32}
            else:
                closure_kwargs = {"max_iters": 2}
            repair = local_placement_closure(
                object_name,
                predicate=effective_predicate,
                target_name=target_name or None,
                require_drawer_closing_clearance=(
                    bool(require_drawer_closing_clearance) and not bool(used_drawer_release)
                ),
                **closure_kwargs,
            )
            print("Local placement closure result:", repair)
            satisfied = bool(isinstance(repair, dict) and repair.get("relation_satisfied"))
            if isinstance(repair, dict) and repair.get("drawer_closing_clearance") is not None:
                drawer_clearance = repair.get("drawer_closing_clearance")
    task_status = release.get("task_status", {}) if isinstance(release, dict) else {}
    if isinstance(repair, dict):
        task_status = repair.get("task_status", task_status)
    if (
        used_drawer_release
        and bool(require_drawer_closing_clearance)
        and not isinstance(drawer_precondition_failure, dict)
    ):
        def _stage_drawer_clearance_from_relation(relation, min_front_clearance=0.110, min_lateral_clearance=0.008):
            try:
                local_delta = relation.get("local_delta")
                site_size_abs = relation.get("site_size_abs", relation.get("site_size"))
                if local_delta is None or site_size_abs is None or len(local_delta) < 3 or len(site_size_abs) < 3:
                    return {"has_clearance": False, "failure_reasons": ["missing_local_geometry"]}
                front_clearance = float(local_delta[1]) + abs(float(site_size_abs[1]))
                lateral_clearance = min(
                    float(local_delta[0]) + abs(float(site_size_abs[0])),
                    abs(float(site_size_abs[0])) - float(local_delta[0]),
                )
                failure_reasons = []
                if front_clearance < float(min_front_clearance):
                    failure_reasons.append("insufficient_front_clearance_for_later_close")
                if lateral_clearance < float(min_lateral_clearance):
                    failure_reasons.append("insufficient_lateral_clearance_for_later_close")
                return {
                    "has_clearance": not failure_reasons,
                    "failure_reasons": failure_reasons,
                    "front_clearance": front_clearance,
                    "lateral_clearance": lateral_clearance,
                    "min_front_clearance": float(min_front_clearance),
                    "min_lateral_clearance": float(min_lateral_clearance),
                    "local_delta": local_delta,
                    "site_size_abs": site_size_abs,
                    "in": relation.get("in"),
                }
            except Exception as clearance_exc:
                return {"has_clearance": False, "failure_reasons": ["clearance_eval_error"], "error": repr(clearance_exc)}

        final_drawer_relation = {}
        if isinstance(repair, dict):
            candidate_relation = repair.get("final_relation") or repair.get("relation") or {}
            if isinstance(candidate_relation, dict):
                final_drawer_relation = candidate_relation
        if not final_drawer_relation and isinstance(release, dict):
            candidate_relation = release.get("final_relation") or release.get("relation") or {}
            if isinstance(candidate_relation, dict):
                final_drawer_relation = candidate_relation
        if final_drawer_relation:
            try:
                min_front = drawer_min_front_clearance_for_repair
                if min_front is None and isinstance(drawer_clearance, dict):
                    min_front = float(drawer_clearance.get("min_front_clearance"))
                if min_front is None:
                    min_front = 0.085 if "bottom" in str(target_name).lower() else 0.065
                verified_drawer_clearance = _stage_drawer_clearance_from_relation(
                    final_drawer_relation,
                    min_front_clearance=float(min_front),
                )
                drawer_clearance = verified_drawer_clearance
                if not bool(verified_drawer_clearance.get("has_clearance", False)):
                    print("Final drawer placement still lacks close-clearance margin; preserving In/contact and deferring to drawer close stage.", verified_drawer_clearance)
            except Exception as clearance_gate_exc:
                print("Final drawer clearance gate failed:", repr(clearance_gate_exc))
                satisfied = False
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    global_repair = None
    if is_final_stage and satisfied and (not completed) and reward < 1.0:
        try:
            goal_diag = get_bddl_goal_diagnostics()
            all_goals = bool(isinstance(goal_diag, dict) and goal_diag.get("all_goals_satisfied", False))
            placement_goals = goal_diag.get("goal_state", []) if isinstance(goal_diag, dict) else []
            has_protected_placement_goal = bool(
                sum(
                    1
                    for goal in placement_goals
                    if (
                        isinstance(goal, dict)
                        and len(goal.get("raw", [])) == 3
                        and str((goal.get("raw") or [""])[0]).lower() in ('in', 'on')
                        and bool(goal.get("satisfied", False))
                    )
                )
                >= 1
            )
        except Exception as goal_exc:
            print("Global BDDL diagnostics unavailable after final release:", goal_exc)
            all_goals = False
            has_protected_placement_goal = False
        if not all_goals and has_protected_placement_goal:
            print("Skipping global physical repair to preserve already satisfied placement goals.")
        elif not all_goals:
            print("Final release completed locally but global BDDL goal is not complete; entering global physical repair.")
            global_repair = global_bddl_goal_repair(max_passes=2, binary_max_iters=1, settle_steps=35)
            print("Global BDDL repair after final release:", global_repair)
            if isinstance(global_repair, dict):
                task_status = global_repair.get("task_status", task_status)
                reward = float(task_status.get("reward", reward) or reward)
                completed = bool(task_status.get("task_completed", completed))
    status = "satisfied" if satisfied or completed or reward >= 1.0 else "needs_repair"
    closure_near = repair.get("near_target", None) if isinstance(repair, dict) else None
    closure_xy = repair.get("xy_distance_to_target", None) if isinstance(repair, dict) else None
    closure_relation = None
    repair_reason_for_evidence = None
    needs_replan_deeper_held_release = False
    needs_repick = bool(isinstance(release, dict) and release.get("needs_repick", False))
    if isinstance(repair, dict):
        repair_reason_for_evidence = repair.get("reason")
        needs_replan_deeper_held_release = bool(repair.get("needs_replan_deeper_held_release", False))
        needs_repick = bool(needs_repick or repair.get("needs_repick", False))
        relation_for_evidence = repair.get("final_relation") or repair.get("initial_relation") or {}
        if isinstance(relation_for_evidence, dict):
            closure_relation = {
                "satisfied": relation_for_evidence.get("satisfied"),
                "on": relation_for_evidence.get("on"),
                "in": relation_for_evidence.get("in"),
                "contact": relation_for_evidence.get("contact"),
                "local_delta": relation_for_evidence.get("local_delta"),
                "site_size_abs": relation_for_evidence.get("site_size_abs", relation_for_evidence.get("site_size")),
                "xy_distance": relation_for_evidence.get("xy_distance"),
            }
    rack_summary = release.get("summary", None) if used_rack_release and isinstance(release, dict) else None
    rack_evidence = ""
    if isinstance(rack_summary, dict):
        final = rack_summary.get("final", {})
        phase_diags = rack_summary.get("phase_diags", {})
        compact_phases = {}
        for phase_name in ("pre_open", "pre_release_rail_kiss", "release_open_no_retreat", "post_release_settle", "skip_micro_lift_after_unsatisfied_release"):
            diag = phase_diags.get(phase_name, {}) if isinstance(phase_diags, dict) else {}
            if isinstance(diag, dict):
                compact_phases[phase_name] = {
                    "on": diag.get("on"),
                    "in_xy": diag.get("in_xy"),
                    "in_z": diag.get("in_z"),
                    "contact": diag.get("contact"),
                    "local_delta": diag.get("local_delta"),
                    "site_size": diag.get("site_size"),
                }
        rack_evidence = ", rack_summary=" + str({"final": final, "phases": compact_phases})
    global_evidence = ""
    if isinstance(global_repair, dict):
        global_evidence = ", global_repair=" + str({
            "success": global_repair.get("success"),
            "reason": global_repair.get("reason"),
            "record_count": len(global_repair.get("records", [])) if isinstance(global_repair.get("records"), list) else None,
            "task_status": global_repair.get("task_status"),
        })
    microwave_evidence = ""
    if used_microwave_release and isinstance(release, dict):
        final_relation = release.get("final_relation", {})
        if isinstance(final_relation, dict):
            microwave_evidence = ", microwave_relation=" + str({
                "satisfied": final_relation.get("satisfied"),
                "in": final_relation.get("in"),
                "contact": final_relation.get("contact"),
                "local_delta": final_relation.get("local_delta"),
                "site_size_abs": final_relation.get("site_size_abs", final_relation.get("site_size")),
            })
        microwave_records = release.get("records", [])
        if isinstance(microwave_records, list):
            microwave_phase_tail = []
            for item in microwave_records[-6:]:
                if not isinstance(item, dict):
                    continue
                entry = {"phase": item.get("phase")}
                seating = item.get("seating")
                if isinstance(seating, dict):
                    subactions = []
                    for rec in seating.get("records", [])[-3:]:
                        if not isinstance(rec, dict):
                            continue
                        subactions.append({
                            "action": rec.get("action"),
                            "z_high_margin": rec.get("z_high_margin"),
                            "improved_z": rec.get("improved_z"),
                        })
                    entry["seating_actions"] = subactions
                microwave_phase_tail.append(entry)
            microwave_evidence += ", microwave_record_phases=" + str(microwave_phase_tail)
    evidence = (
        f"relation_satisfied={satisfied}, microwave_near_inside_z_only={microwave_near_inside_z_only}, preserved_drawer_z_only={preserved_drawer_z_only}, require_drawer_closing_clearance={require_drawer_closing_clearance}, "
        f"drawer_clearance={drawer_clearance}, reward={reward:.3f}, task_completed={completed}, "
        f"closure_near={closure_near}, closure_xy={closure_xy}, "
        f"closure_relation={closure_relation}, repair_reason={repair_reason_for_evidence}, "
        f"needs_replan_deeper_held_release={needs_replan_deeper_held_release}, needs_repick={needs_repick}, "
        f"object={object_name}, target={target_name}, predicate={effective_predicate}{rack_evidence}{microwave_evidence}{global_evidence}"
    )
    if status == "satisfied":
        repair_hint = ""
    elif used_microwave_release:
        repair_hint = (
            "Microwave insertion is xy-valid but still above the heating-region z bound; preserve this failure evidence and retry only z/low-release correction."
            if microwave_near_inside_z_only
            else "Microwave insertion failed; keep repair local to microwave insertion/seating before any close stage."
        )
    elif used_drawer_release and isinstance(drawer_precondition_failure, dict):
        repair_hint = "Drawer is closed or too shallow for insertion; reopen/pull the drawer farther before retrying target-object placement."
    elif used_drawer_release and isinstance(drawer_clearance, dict) and not bool(drawer_clearance.get("has_clearance", False)):
        repair_hint = "Drawer front clearance is insufficient after In/contact; stop bounded local seating and re-grasp the bowl for a deeper held drawer release before closing."
    elif used_drawer_release:
        repair_hint = "Drawer insertion failed; preserve the failed relation evidence and retry with a held target-object insertion instead of unheld local closure."
    elif used_thin_caddy_release and isinstance(release, dict) and bool(release.get("needs_repick", False)):
        repair_hint = "Object is not verified-held before caddy release; return control to the pick_lift stage, reacquire the object, then redo guarded transport and low release."
    else:
        repair_hint = "Release/local closure did not satisfy target relation; continue bounded local physical closure before any re-pick."
except Exception as exc:
    status = "needs_repair"
    evidence = f"low release fast-path exception: {exc}"
    repair_hint = "Retry release after checking target grounding and held object state."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair_hint,
}, default=str))
