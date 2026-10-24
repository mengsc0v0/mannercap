"""Step 6 (draft): LIBERO-10 capability matrix + request table + split.  Facts (objects, goal, init predicates, init-state
count) are read from the benchmark BDDL via s6_libero10_extract.py output; primitive coverage and applicable manners are
ASSESSMENTS (status says whether anything was executed).  Requests are template-generated here; the LLM composition /
independent semantic verification is the formal Step 6 and is not claimed by this file."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "reports" / "step6_libero10_spec_20261025"
TASKS = {t["id"]: t for t in json.loads((D / "libero10_tasks.json").read_text())}

# stage instance kinds: pick_place(obj->target), close(articulated), turn_on(knob)
def pp(o, t, cov, note=""):
    return {"kind": "pick_place", "object": o, "target": t, "coverage": cov, "note": note}

SPEC = {
 0: dict(group="A 篮子多对象", stages=[pp("alphabet_soup_1", "basket_1_contain_region", "reuse_unverified", "罐体圆柱，抓取绑定/接触几何需验证"),
                                      pp("tomato_sauce_1", "basket_1_contain_region", "reuse_unverified", "同上")]),
 1: dict(group="A 篮子多对象", stages=[pp("cream_cheese_1", "basket_1_contain_region", "verified", "Step3-5 开发验证"),
                                      pp("butter_1", "basket_1_contain_region", "verified", "同上")]),
 7: dict(group="A 篮子多对象", stages=[pp("alphabet_soup_1", "basket_1_contain_region", "reuse_unverified", "罐体圆柱"),
                                      pp("cream_cheese_1", "basket_1_contain_region", "reuse_unverified", "对象已在 T1 验证，场景不同")]),
 4: dict(group="A 双杯/杯+布丁", stages=[pp("porcelain_mug_1", "plate_1", "adapt", "放到盘上(On)，杯体抓取位姿与放置高度需新定义"),
                                       pp("white_yellow_mug_1", "plate_2", "adapt", "同上；场景含干扰杯 red_coffee_mug_1")]),
 6: dict(group="A 双杯/杯+布丁", stages=[pp("porcelain_mug_1", "plate_1", "adapt", "放到盘上"),
                                       pp("chocolate_pudding_1", "living_room_table_plate_right_region", "adapt", "目标是桌面区域而非物体")]),
 8: dict(group="A 重复抓放", stages=[pp("moka_pot_1", "flat_stove_1_cook_region", "adapt", "摩卡壶抓取位姿；BDDL 初态声明炉灶已开(Turnon)，需运行时核实"),
                                    pp("moka_pot_2", "flat_stove_1_cook_region", "adapt", "同上")]),
 5: dict(group="B 空间约束放置", stages=[pp("black_book_1", "desk_caddy_1_back_contain_region", "adapt", "薄物体 + 隔间约束，接近/对准速度为主；力度指标适用性待验证")]),
 3: dict(group="C 抓放+关节", stages=[pp("akita_black_bowl_1", "white_cabinet_1_bottom_region", "adapt", "放入初态已打开的抽屉(BDDL: Open)"),
                                     {"kind": "close_articulated", "object": "white_cabinet_1_bottom_region", "target": None, "coverage": "missing", "note": "缺少关抽屉(推)原语"}]),
 9: dict(group="C 抓放+关节", stages=[pp("white_yellow_mug_1", "microwave_1_heating_region", "adapt", "放入初态已打开的微波炉(BDDL: Open)"),
                                     {"kind": "close_articulated", "object": "microwave_1", "target": None, "coverage": "missing", "note": "缺少关门原语"}]),
 2: dict(group="C 抓放+关节", stages=[{"kind": "turn_on", "object": "flat_stove_1", "target": None, "coverage": "missing", "note": "缺少旋钮/开关原语"},
                                     pp("moka_pot_1", "flat_stove_1_cook_region", "adapt", "放到炉面区域")]),
}
# applicable manner axes (assessment) -> evaluation window
AXES = {
 "hold_force": {"window": "lift[TRIM:] ∪ transport 事件窗(接触事件接口)", "metric": "两指法向力合计(N)，逐物体标定带",
                 "status": "T1 已验证(v1 联合 28/32)；其它对象需逐物体标定"},
 "stage_speed": {"window": "各阶段实例步段(approach/transport/place)", "metric": "末端速度均值/峰值", "status": "StagePlan.v 已可控；响应未标定"},
 "articulation_speed": {"window": "关节运动阶段(到位前减速段单独计)", "metric": "关节/把手速度", "status": "需新原语后才可测"},
}
for t, sp in SPEC.items():
    sp["axes"] = ["hold_force", "stage_speed"] if all(s["kind"] == "pick_place" for s in sp["stages"]) else ["hold_force", "stage_speed", "articulation_speed"]
NAMES = {"alphabet_soup_1": "alphabet soup", "tomato_sauce_1": "tomato sauce", "cream_cheese_1": "cream cheese box", "butter_1": "butter",
         "porcelain_mug_1": "white mug", "white_yellow_mug_1": "yellow and white mug", "chocolate_pudding_1": "chocolate pudding",
         "moka_pot_1": "moka pot", "moka_pot_2": "second moka pot", "black_book_1": "book", "akita_black_bowl_1": "black bowl"}
ART = {"white_cabinet_1_bottom_region": "drawer", "microwave_1": "microwave door", "flat_stove_1": "stove knob"}

def reqs():
    out = []
    def add(t, axis, level, text, obj, stage, ok_now, note="", kind="following"):
        out.append({"request_id": f"L10-T{t}-{len(out):03d}", "task_id": t, "task_text": TASKS[t]["language"], "manner_axis": axis,
                    "level": level, "request_text": text, "object": obj, "stage_instance": stage, "target_band": "待标定(Step 6/7)" if axis != "hold_force" or t != 1 else "configs/manner_calibration_libero_t1.v*.json",
                    "eval_window": AXES[axis]["window"], "executable_now": ok_now, "note": note, "request_kind": kind})
    for t, sp in SPEC.items():
        pps = [s for s in sp["stages"] if s["kind"] == "pick_place"]
        ok_now = all(s["coverage"] == "verified" for s in pps) and all(s["kind"] == "pick_place" for s in sp["stages"])
        names = [NAMES[s["object"]] for s in pps]
        if len(pps) == 2:
            a, b = pps
            add(t, "hold_force", "A gentle / B firm", f"{TASKS[t]['language']}; hold the {names[0]} gently and the {names[1]} firmly", f"{a['object']},{b['object']}", "grasp+lift+transport per object", ok_now)
            add(t, "hold_force", "A firm / B gentle", f"{TASKS[t]['language']}; hold the {names[0]} firmly and the {names[1]} gently", f"{a['object']},{b['object']}", "grasp+lift+transport per object", ok_now)
        else:
            for lv in ("gentle", "firm"):
                add(t, "hold_force", lv, f"{TASKS[t]['language']}; hold the {names[0]} {lv}ly", pps[0]["object"], "grasp+lift+transport", ok_now)
        for o_i, s in enumerate(pps[:1]):
            add(t, "stage_speed", "approach fast / place slow", f"{TASKS[t]['language']}; approach the {names[0]} fast, place it slowly", s["object"], "approach, place", False, "速度响应未标定")
            add(t, "stage_speed", "approach slow / place fast", f"{TASKS[t]['language']}; approach the {names[0]} slowly, place it fast", s["object"], "approach, place", False, "速度响应未标定")
        arts = [s for s in sp["stages"] if s["kind"] != "pick_place"]
        for s in arts:
            for lv in ("slow", "fast"):
                add(t, "articulation_speed", lv, f"{TASKS[t]['language']}; do the {ART[s['object']]} step {lv}", s["object"], s["kind"], False, "需新原语")
    return out

PROBES = [  # must be rejected / clarified, never silently executed
 (4, "put the white mug on the left plate and put the yellow and white mug on the right plate; hold the mug gently", "ambiguous: two mugs named"),
 (6, "put the white mug on the plate and put the chocolate pudding to the right of the plate; hold it firmly", "ambiguous: 'it' unresolved"),
 (9, "put the yellow and white mug in the microwave and close it; hold the mug gently", "ambiguous only if distractor mug referenced; check object phrase mapping"),
 (3, "put the black bowl in the bottom drawer of the cabinet and close it; close the drawer with gentle hold force", "unsupported: hold force is not defined for drawer closing"),
 (1, "put both the cream cheese box and the butter in the basket; touch the basket gently", "unsupported: gentle contact (transient) not measured"),
 (1, "put both the cream cheese box and the butter in the basket; move slowly", "under-specified: stage not given"),
 (0, "put both the alphabet soup and the tomato sauce in the basket; hold the ketchup gently", "object not in task goal (distractor)"),
 (5, "pick up the book and place it in the back compartment of the caddy; hold it with 5 N", "numeric force: not a calibrated level"),
]

def main():
    R = reqs()
    # automatic traceability checks
    chk = {"object_in_bddl": 0, "task_text_matches_bddl": 0, "stage_instance_present": 0}
    for r in R:
        objs = r["object"].split(",")
        chk["object_in_bddl"] += all(o in TASKS[r["task_id"]]["objects"] or o in TASKS[r["task_id"]]["goal"] or o in TASKS[r["task_id"]]["fixtures"] or o in TASKS[r["task_id"]]["init"] for o in objs)
        chk["task_text_matches_bddl"] += r["request_text"].startswith(TASKS[r["task_id"]]["language"])
        chk["stage_instance_present"] += bool(r["stage_instance"])
    split = {"calibration": list(range(8, 14)), "dev": list(range(0, 8)), "validation": list(range(14, 22)), "reserve": list(range(22, 40)), "holdout": list(range(40, 50))}
    used = {"1": {"dev(0-7)": "已用于 Step 3-5 开发/定位(已成开发用例)", "calibration(8-13)": "v1 标定已用", "validation(14-21)": "Step 5 v2 正式验证将使用", "holdout(40-49)": "未使用"}}
    cap = {str(t): {"id": t, "name": TASKS[t]["name"], "language": TASKS[t]["language"], "bddl_objects": TASKS[t]["objects"], "bddl_goal": TASKS[t]["goal"],
                    "bddl_init_articulation_predicates": re.findall(r"\((?:Open|Close|Turnon|Turnoff)\s[^)]*\)", TASKS[t]["init"]),
                    "n_init_states": TASKS[t]["n_init_states"], "group": SPEC[t]["group"], "stages": SPEC[t]["stages"], "applicable_axes": SPEC[t]["axes"]} for t in sorted(SPEC)}
    (D / "libero10_capability_matrix.json").write_text(json.dumps({"axes": AXES, "tasks": cap, "split": split, "split_usage": used}, indent=1, ensure_ascii=False))
    (D / "libero10_requests_draft.json").write_text(json.dumps({"requests": R, "ambiguity_probes": [{"task_id": t, "text": x, "expected": e} for t, x, e in PROBES],
                                                                 "traceability_auto_checks": {k: [v, len(R)] for k, v in chk.items()}}, indent=1, ensure_ascii=False))
    print("tasks", len(cap), "requests", len(R), "probes", len(PROBES), chk)
    cov = {}
    for t, c in cap.items():
        for s in c["stages"]:
            cov.setdefault(s["coverage"], []).append(f"T{t}:{s['kind']}")
    for k, v in cov.items():
        print(k, len(v), v)
    print("executable_now requests", sum(r["executable_now"] for r in R), "/", len(R))

main()
