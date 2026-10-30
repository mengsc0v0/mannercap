"""Step 7: LIBERO basket pick-place tasks as DATA.  One spec per task; program/plan/session are built by the same shared
templates and executor as task 1 (s3_libero_task).  Anything a task needs beyond this table (a new field, a changed
constant, a code branch) is a shared-code or task-specific change and must be logged as such."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synthesis_v2.libero_backend import LiberoSession  # noqa: E402
from synthesis_v2.native_producer import GripPlan, StagePlan  # noqa: E402
from synthesis_v2.program_executor import ProgramPlan  # noqa: E402
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances  # noqa: E402

BASKET_RELEASE_DZ, BASKET_TRANSPORT_DZ = 0.14, 0.175   # measured on task 1's basket (s3_libero_task); reused as-is, same basket asset


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    objects: tuple            # ((role, backend object), ...) in program order
    target: tuple             # (role, backend object); for per-object targets: the first one (kept for single-target callers)
    suite: str = "libero_10"
    targets: tuple = ()       # optional per-object targets ((role, backend object), ...) aligned with ``objects``
    grasp_grid: tuple = ()     # optional wider grasp-candidate ranges (yaws_deg, laterals_m, dzs_m); () = executor default grid
    grasp_reference: tuple = ()  # optional per-object grasp reference aligned with ``objects`` (dict(rel_obj, hand_yaw_rel_deg) or None)
    container_dz: bool = True  # True: basket release/transport heights; False: executor defaults (STACK_PLACE_DZ / LIFT_DZ)
    # Step 14 (B): per-task constants for a target given as a scene region (``@site:<name>``).  Keys: target_offset_site (x, y, z in
    # the site frame), target_yaw_deg / target_yaw_symmetry_deg (object yaw in the site frame), lift_dz, transport_dz, place_dz.
    target_params: dict = field(default_factory=dict)


SPECS = {
    1: TaskSpec(1, (("cream_cheese", "cream_cheese_1"), ("butter", "butter_1")), ("basket", "basket_1")),
    0: TaskSpec(0, (("alphabet_soup", "alphabet_soup_1"), ("tomato_sauce", "tomato_sauce_1")), ("basket", "basket_1")),
    7: TaskSpec(7, (("alphabet_soup", "alphabet_soup_1"), ("cream_cheese", "cream_cheese_1")), ("basket", "basket_1")),
    # Step 14 (T5 pre-check): config-only entry.  One object, target = the desk caddy BODY (not its back compartment);
    # executor defaults for carry/release heights.  Whatever this cannot express is a gap, not tuned here.
    5: TaskSpec(5, (("book", "black_book_1"),), ("caddy_back", "@site:desk_caddy_1_back_contain_region"), container_dz=False,
                # Official goal In(black_book_1, desk_caddy_1_back_contain_region).  Step 15: the region is "the book fits in here", so the
                # book's COLLISION centre (not its body origin, 1.2 cm away in y) is aligned over the site (target_align); the height is the
                # demo_0 book-origin height at its first environment success (region frame z -0.0533, i.e. 5.3 cm above the site centre).
                # Yaw 0 mod 180: book thin axis along the site x axis.  place_tol_m: the measured lateral margin between the book's
                # collision box and the compartment walls is ~6 mm on one side (reports/step15_t5_place/fit_margin.json), so the place pose
                # is accepted only within 4 mm (the executor default 8 mm would not guarantee the book is inside the walls).
                # lift/transport heights from the scene (read from the fixture collision geoms, not searched): carried book bottom =
                # compartment end-wall top 1.054 + 0.02 margin -> centre 1.141 -> lift 0.26 above the book's start, transport 0.17 above the site.
                target_params={"target_offset_site": (0.0, 0.0, -0.0533), "target_align": "collision_center", "target_yaw_deg": 0.0,
                               "target_yaw_symmetry_deg": 180.0, "lift_dz": 0.26, "transport_dz": 0.17, "place_dz": 0.0, "place_tol_m": 0.004},
                # Step 14 (C): ONE ordinary official demonstration, used as is (no averaging, no per-init choice).  Pre-registered
                # rule: replay-successful in two independent replays, lowest index -> demo_0 (reports/step14_t5_onboard/demo_replay).
                # hand minus book origin in the book yaw frame at the demo's last close + hand yaw vs the executor's yaw-aligned quat.
                grasp_reference=({"rel_obj": [0.0084, -0.0123, 0.1209], "hand_yaw_rel_deg": -4.1},)),
    # Step 9: first task whose targets are two different objects (plates) and that has no container
    4: TaskSpec(4, (("white_mug", "porcelain_mug_1"), ("yellow_mug", "white_yellow_mug_1")), ("plate_left", "plate_1"),
                targets=(("plate_left", "plate_1"), ("plate_right", "plate_2")), container_dz=False,
                grasp_grid=((0, 90, -90, 45, -45, 135, -135, 180, 30, -30, 60, -60, 120, -120, 150, -150), (0.0, 0.03, -0.03, 0.04, -0.04, 0.05, -0.05, 0.06, -0.06),
                            (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)),
                # reference = median over official-demo grasps (replay-successful demos; reports/step9_t4_onboard/ref_grasp.json): hand offset in the
                # object yaw frame + hand yaw relative to the executor's yaw-aligned quaternion.  Used as is (no tuning).
                grasp_reference=({"rel_obj": [-0.0319, -0.013, 0.101], "hand_yaw_rel_deg": 10.3},
                                 {"rel_obj": [-0.0572, -0.0163, 0.0762], "hand_yaw_rel_deg": 1.8})),
}


# Step 14 (T5): no grasp grid of its own any more.  The pre-check showed that widening the grid (T4's grid) still never reached the
# grasp band; T5 takes its grasp from one ordinary official demonstration (grasp_reference in its TaskSpec).


# Named grasp-closure configurations (algorithm settings, saved here rather than patched at call sites).
GRIP_CONFIGS = {
    "baseline": GripPlan(3.0, 9.0),                                          # historical policy
    "settle": GripPlan(3.0, 9.0, settle_after_open=1),                       # wait one step after a release pulse, then re-judge
    "settle_reclose": GripPlan(3.0, 9.0, settle_after_open=1, reclose_max=2),  # + bounded re-close when re-checked force < lo
    "libero_t1": GripPlan(8.0, 40.0),                                        # force band already validated on task 1 (s3_libero_validate.GRIP)
}


def make_program(t: int) -> TaskProgram:
    sp = SPECS[t]
    tg = sp.targets or tuple(sp.target for _ in sp.objects)
    roles = {r: n for r, n in sp.objects}; roles.update({r: n for r, n in tg})
    inst = chain(*[pick_place_instances(r, tg[k][0]) for k, (r, _) in enumerate(sp.objects)])
    if sp.container_dz:
        inst = tuple(replace(i, params={**i.params, "place_dz": BASKET_RELEASE_DZ}) if i.primitive == "place" else i for i in inst)
        inst = tuple(replace(i, params={**i.params, "transport_dz": BASKET_TRANSPORT_DZ}) if i.primitive == "transport" else i for i in inst)
    if sp.target_params:
        tp = dict(sp.target_params)
        lift = {"lift_dz": tp.pop("lift_dz")} if "lift_dz" in tp else {}
        inst = tuple(replace(i, params={**i.params, **tp}) if i.primitive in ("transport", "place")
                     else replace(i, params={**i.params, **lift}) if i.primitive == "lift" and lift else i for i in inst)
    if sp.grasp_grid:
        gg = {"yaws_deg": list(sp.grasp_grid[0]), "laterals_m": list(sp.grasp_grid[1]), "dzs_m": list(sp.grasp_grid[2])}
        inst = tuple(replace(i, params={**i.params, "grasp_grid": gg}) if i.primitive == "approach" else i for i in inst)
    if sp.grasp_reference:
        ref = {sp.objects[k][0]: g for k, g in enumerate(sp.grasp_reference) if g}
        inst = tuple(replace(i, params={**i.params, "grasp_reference": ref[i.object_role]}) if i.primitive == "approach" and i.object_role in ref else i for i in inst)
    return TaskProgram(f"libero_t{t}_{'basket' if sp.container_dz else 'plates'}_{len(sp.objects)}", "libero", f"libero:{sp.suite}:{t}", roles, inst, source="composed")


def make_plan(t: int, prog: TaskProgram, grip=None, v: float = 0.12, **kw) -> ProgramPlan:
    sp = SPECS[t]; grip = grip or GripPlan(3.0, 9.0)
    return ProgramPlan(prog, tuple(StagePlan(it.label, v) for it in prog.instances),
                       {f"{r}.pick_1.grasp": (grip[r] if isinstance(grip, dict) else grip) for r, _ in sp.objects}, **kw)


def make_session(t: int, init_index: int, prog: TaskProgram, render: bool = False) -> LiberoSession:
    sp = SPECS[t]
    return LiberoSession(t, init_index, suite=sp.suite, render=render, default_target=prog.objects[sp.objects[0][0]])
