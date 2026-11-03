"""Step 30: manner request -> LIBERO control quantities -> per-clause audit -> bounded repair.

Pure module (no simulator).  The language side (LLM intake + admit) produces bound clauses; this module turns them
into per-stage speed commands and per-object grip closure configurations, judges every clause on the executed
episode, and proposes the next commands when a clause failed.  Repair never leaves the requested band: a request
that cannot be met inside its band is reported as infeasible, never silently relaxed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

MANNER_LIBERO_VERSION = "manner_libero.v2"

# measured-quantity bands (pre-registered in reports/step30_manner_gen/protocol.json)
SPEED_BAND = {"slow": (0.0, 0.09), "normal": (0.09, 0.17), "fast": (0.17, 1.0)}
GRIP_BAND = {"gentle": (8.0, 16.0), "normal": (16.0, 24.0), "firm": (24.0, 34.0)}
GRIP_BAND_NAME = {"gentle": "R1", "normal": "R2", "firm": "R3"}
# commanded stage speed per level; repair walks the ladder in the direction of the miss, never across levels
SPEED_LADDER = {"slow": (0.06, 0.05, 0.04), "normal": (0.12, 0.10, 0.15), "fast": (0.24, 0.32, 0.40)}
DEFAULT_V = 0.12                 # unbound stage
DEFAULT_GRIP = (1.0, 100.0)      # unbound object (executor default closure)
CARRY = ("lift", "transport")
# Carry-clearance repair (step 31): when an execution fails (task or contact) and the failure is attributed to one
# object's carry (unexcused robot contact during its lift/transport, or another tracked object displaced while it is
# carried), that object's lift/transport heights are raised by CLEAR_STEP, at most CLEAR_MAX in total.  Manner
# commands (speed, grip) are left unchanged by this action.
CLEAR_STEP, CLEAR_MAX = 0.03, 0.06


class MannerRefused(ValueError):
    pass


@dataclass(frozen=True)
class MannerRequest:
    speed: Mapping[str, str]                 # instance_id -> level
    grip: Mapping[str, str]                  # object role -> level
    clauses: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"speed": dict(self.speed), "grip": dict(self.grip), "clauses": list(self.clauses)}


def request_from_semantics(program, semantics) -> MannerRequest:
    """Admitted bindings -> request.  Speed binds stage instances, grip binds the grasp instance's object."""
    speed: dict[str, str] = {}
    grip: dict[str, str] = {}
    clauses = []
    for b in semantics.bindings:
        if b.dimension not in ("speed", "grip"):
            raise MannerRefused(f"unsupported_dimension:{b.dimension}")
        for iid in b.instance_ids:
            it = program.instance(iid)
            if b.dimension == "speed":
                if it.primitive == "grasp":
                    raise MannerRefused(f"speed_on_grasp:{iid}")
                if speed.get(iid, b.level) != b.level:
                    raise MannerRefused(f"conflict:{iid}")
                speed[iid] = b.level
            else:
                if it.primitive != "grasp":
                    raise MannerRefused(f"grip_on_non_grasp:{iid}")
                if grip.get(it.object_role, b.level) != b.level:
                    raise MannerRefused(f"conflict:{it.object_role}")
                grip[it.object_role] = b.level
        clauses.append({"dimension": b.dimension, "level": b.level, "instance_ids": list(b.instance_ids),
                        "clause": b.clause})
    return MannerRequest(speed, grip, tuple(clauses))


def carry_level(program, request: MannerRequest, role: str) -> Optional[str]:
    """Bound speed level of the object's transport (the carry), or None."""
    for it in program.instances:
        if it.object_role == role and it.primitive == "transport" and it.instance_id in request.speed:
            return request.speed[it.instance_id]
    return None


def initial_commands(program, request: MannerRequest, choose_grip: Callable[[str, str, float], tuple],
                     clear: Optional[Mapping[str, float]] = None) -> dict:
    """speed: instance_id -> v_max; grip: role -> closure (lo, hi).

    choose_grip(role, grip_level, carry_v) -> (force, carry v | None).  A grip clause may move the object's carry
    speed (lift + transport) only where no speed clause binds it, or inside the bound clause's band."""
    sp = {it.instance_id: (SPEED_LADDER[request.speed[it.instance_id]][0] if it.instance_id in request.speed
                           else DEFAULT_V)
          for it in program.instances if it.primitive != "grasp"}
    gr = {}
    for it in program.instances:
        if it.primitive == "grasp":
            role = it.object_role
            if role in request.grip:
                f, v = choose_grip(role, request.grip[role], carry_v(program, sp, role))
                gr[role] = tuple(f)
                set_carry(program, request, sp, role, v)
            else:
                gr[role] = DEFAULT_GRIP
    return {"speed": sp, "grip": gr, "clear": {r: float(v) for r, v in (clear or {}).items() if v}}


def allowed_carry(program, request: MannerRequest, role: str, grid: tuple) -> list:
    """Carry speeds a grip decision may use for this object: the grid, restricted to the bound carry band."""
    lvl = carry_level(program, request, role)
    if lvl is None:
        return sorted(grid)
    lo, hi = SPEED_BAND[lvl]
    return sorted(v for v in set(grid) | set(SPEED_LADDER[lvl]) if lo <= v <= hi)


def set_carry(program, request: MannerRequest, speed_cmd: dict, role: str, v: Optional[float]) -> list:
    moved = []
    if v is None:
        return moved
    for it in program.instances:
        if it.object_role == role and it.primitive in CARRY:
            lvl = request.speed.get(it.instance_id)
            if (lvl is None or SPEED_BAND[lvl][0] <= v <= SPEED_BAND[lvl][1]) and speed_cmd[it.instance_id] != v:
                speed_cmd[it.instance_id] = v
                moved.append(it.instance_id)
    return moved


def carry_v(program, speed_cmd: Mapping[str, float], role: str) -> float:
    for it in program.instances:
        if it.object_role == role and it.primitive == "transport":
            return float(speed_cmd[it.instance_id])
    return DEFAULT_V


def audit(request: MannerRequest, measured: Mapping[str, Any]) -> dict:
    """measured = {task_ok, contact_clean, stage_speed: {iid: median m/s | None},
                   objects: {role: {hold_min, hold_max, hold_mean, stable, valid}}}"""
    rows = []
    for role, lvl in request.grip.items():
        lo, hi = GRIP_BAND[lvl]
        o = measured["objects"].get(role) or {}
        if not o.get("valid") or not o.get("stable") or o.get("hold_mean") is None:
            cause = "slip_or_drop"
        elif o["hold_max"] > hi:
            cause = "force_above_band"
        elif o["hold_min"] < lo:
            cause = "force_below_band"
        else:
            cause = None
        rows.append({"dimension": "grip", "target": role, "level": lvl, "band": [lo, hi], "pass": cause is None,
                     "cause": cause, "measured": {k: o.get(k) for k in ("hold_min", "hold_mean", "hold_max")}})
    for iid, lvl in request.speed.items():
        lo, hi = SPEED_BAND[lvl]
        v = measured["stage_speed"].get(iid)
        cause = "speed_unmeasured" if v is None else ("speed_below_band" if v < lo else
                                                      "speed_above_band" if v > hi else None)
        rows.append({"dimension": "speed", "target": iid, "level": lvl, "band": [lo, hi], "pass": cause is None,
                     "cause": cause, "measured": v})
    manner_ok = all(r["pass"] for r in rows)
    task_ok, clean = bool(measured["task_ok"]), bool(measured["contact_clean"])
    return {"clauses": rows, "manner_ok": manner_ok, "task_ok": task_ok, "contact_clean": clean,
            "joint_pass": manner_ok and task_ok and clean}


def next_commands(program, request: MannerRequest, cmd: dict, aud: dict, tried: list,
                  repair_force: Callable[[str, str, tuple, str, float, list], Optional[tuple]],
                  carry_evidence: Optional[Mapping[str, list]] = None) -> tuple[Optional[dict], list]:
    """One repair step on every failed clause.  Returns (new_cmd | None, actions).

    repair_force(role, grip_level, prev_force, cause, carry_v, tried_forces) -> (new force, new carry v | None) or None.
    None as new_cmd = nothing left inside the requested bands (or the failure is not a manner failure)."""
    failed = [r for r in aud["clauses"] if not r["pass"]]
    if aud["joint_pass"]:
        return None, [{"action": "none", "why": "joint_pass"}]
    new = {"speed": dict(cmd["speed"]), "grip": dict(cmd["grip"]), "clear": dict(cmd.get("clear") or {})}
    acts = []
    ev = {r: e for r, e in (carry_evidence or {}).items() if e}
    if not (aud["task_ok"] and aud["contact_clean"]):
        for role in sorted(ev):
            cur = float(new["clear"].get(role, 0.0))
            if cur + CLEAR_STEP <= CLEAR_MAX + 1e-9:
                new["clear"][role] = round(cur + CLEAR_STEP, 3)
                acts.append({"action": "clearance", "target": role, "from": cur, "to": new["clear"][role],
                             "cause": list(ev[role])[:3]})
    if not failed and not acts:
        why = "task_failed" if not aud["task_ok"] else "contact_unclean"
        if ev:
            return None, [{"action": "stop", "why": "carry_clearance_exhausted", "target": sorted(ev)}]
        return None, [{"action": "stop", "why": f"{why}_not_a_manner_failure"}]
    for r in failed:
        if r["dimension"] == "speed":
            iid, lad, cur = r["target"], SPEED_LADDER[r["level"]], cmd["speed"][r["target"]]
            if r["cause"] == "speed_below_band":
                nxt = [v for v in lad if v > cur]
            elif r["cause"] == "speed_above_band":
                nxt = [v for v in lad if v < cur]
            else:
                nxt = []
            nxt = sorted(nxt, key=lambda v: abs(v - cur))
            if not nxt:
                return None, acts + [{"action": "stop", "why": "speed_ladder_exhausted", "target": iid}]
            new["speed"][iid] = nxt[0]
            acts.append({"action": "speed", "target": iid, "from": cur, "to": nxt[0], "cause": r["cause"]})
        else:
            role = r["target"]
            prev = tuple(cmd["grip"][role])
            tf = [tuple(t["grip"][role]) for t in tried]
            v0 = carry_v(program, cmd["speed"], role)
            out = repair_force(role, r["level"], prev, r["cause"], v0, tf)
            if out is None:
                return None, acts + [{"action": "stop", "why": "force_candidates_exhausted", "target": role}]
            f, v = out
            new["grip"][role] = tuple(f)
            act = {"action": "grip", "target": role, "from": list(prev), "to": list(f), "cause": r["cause"]}
            moved = set_carry(program, request, new["speed"], role, v)   # only where the stage's own band allows it
            if moved:
                act.update(carry_from=v0, carry_to=v, carry_instances=moved)
            acts.append(act)
    if any(_same(new, t) for t in tried):
        return None, acts + [{"action": "stop", "why": "repeat_of_tried_commands"}]
    return new, acts


def _same(a: dict, b: dict) -> bool:
    return a["speed"] == b["speed"] and {k: tuple(v) for k, v in a["grip"].items()} == \
        {k: tuple(v) for k, v in b["grip"].items()} and \
        {k: v for k, v in (a.get("clear") or {}).items() if v} == {k: v for k, v in (b.get("clear") or {}).items() if v}
