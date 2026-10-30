"""Step 15 (3): the PLACE part of an ordinary official T5 demonstration, in the REGION frame, unaveraged, real 3-D poses.
Replays one demo in the matched env and records, per control step, hand and book pose in the official target-site frame, the
hand pose relative to the book (book frame), the environment's own success flag and the book<->caddy clearance.
Key poses: after-lift / above-region / first-in-region / last step; plus whether the demo releases at all.
usage: s15_place_extract.py DEMO_FILE OUT_DIR DEMO_INDEX..."""
import json, math, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/workdata/cap-x/capx/third_party/LIBERO-PRO")
sys.path.append(LIBERO_ROOT); os.environ.setdefault("MUJOCO_GL", "egl")
import h5py, mujoco  # noqa: E402
from libero import benchmark  # noqa: E402
from libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.utils import get_libero_path  # noqa: E402

FILE, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
T, BOOK, SITE, CADDY = 5, "black_book_1", "desk_caddy_1_back_contain_region", "desk_caddy_1_main"
task = benchmark.get_benchmark_dict()["libero_10"]().get_task(T)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                         camera_heights=128, camera_widths=128, controller="OSC_POSE", control_freq=20)
rd = lambda a: [round(float(x), 4) for x in a]


def pose_in(Rs, ps, p, q_xyzw):   # world pose -> region frame: position and rotation (rotvec-free: xyzw quat + euler deg)
    rot = R.from_matrix(Rs.T) * R.from_quat(q_xyzw)
    return rd(Rs.T @ (p - ps)), rd(rot.as_quat()), rd(rot.as_euler("xyz", degrees=True))


with h5py.File(FILE, "r") as f:
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    for di in [int(x) for x in sys.argv[3:]]:
        g = f["data"][keys[di]]; acts, states = g["actions"][:], g["states"][:]
        env.reset(); obs = env.set_init_state(states[0]); sim = env.env.sim
        m, d = sim.model._model, sim.data._data
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
        sp, sR = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
        bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BOOK + "_main"); cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CADDY)
        bg = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bb and (m.geom_contype[i] or m.geom_conaffinity[i])]
        cg = [i for i in range(m.ngeom) if m.geom_bodyid[i] == cb and (m.geom_contype[i] or m.geom_conaffinity[i])]
        tr, fromto = [], np.zeros(6)
        for k, a in enumerate(acts):
            ep, eq = np.asarray(obs["robot0_eef_pos"], float), np.asarray(obs["robot0_eef_quat"], float)
            bp, bq = np.asarray(obs[BOOK + "_pos"], float), np.asarray(obs[BOOK + "_quat"], float)
            best = (9.0, -1, -1)
            for gi in bg:
                for gj in cg:
                    dist = mujoco.mj_geomDistance(m, d, gi, gj, 0.2, fromto)
                    if dist < best[0]: best = (dist, gi, gj)
            tr.append(dict(k=k, g=float(a[-1]), ep=ep, eq=eq, bp=bp, bq=bq, ok=bool(env.env._check_success()),
                           clr=float(best[0]), clr_caddy_geom=int(best[2])))
            obs, *_ = env.step(a)
        tr.append(dict(k=len(acts), g=0.0, ep=np.asarray(obs["robot0_eef_pos"], float), eq=np.asarray(obs["robot0_eef_quat"], float),
                       bp=np.asarray(obs[BOOK + "_pos"], float), bq=np.asarray(obs[BOOK + "_quat"], float),
                       ok=bool(env.env._check_success()), clr=float("nan"), clr_caddy_geom=-1))
        close = [k for k in range(1, len(acts)) if tr[k]["g"] > 0 and tr[k - 1]["g"] <= 0]
        opens = [k for k in range(1, len(acts)) if tr[k]["g"] <= 0 and tr[k - 1]["g"] > 0]
        kc = close[-1]
        first_ok = next((t["k"] for t in tr if t["ok"]), None)
        z = np.array([t["bp"][2] for t in tr]); kpeak = int(kc + np.argmax(z[kc:]))
        loc = np.array([sR.T @ (t["bp"] - sp) for t in tr])
        above = [k for k in range(kpeak, len(tr)) if abs(loc[k][0]) < 0.0278 and abs(loc[k][1]) < 0.0622]   # xy inside the region
        k_above = above[0] if above else None

        def key(k):
            t = tr[k]; bq = R.from_quat(t["bq"]); hq = R.from_quat(t["eq"])
            bpl, bql, bel = pose_in(sR, sp, t["bp"], t["bq"]); hpl, hql, hel = pose_in(sR, sp, t["ep"], t["eq"])
            by = bq.as_euler("xyz")[2]
            return {"step": k, "grip_cmd": t["g"], "env_success": t["ok"],
                    "book_in_region": {"pos": bpl, "quat_xyzw": bql, "euler_deg": bel},
                    "hand_in_region": {"pos": hpl, "quat_xyzw": hql, "euler_deg": hel},
                    "hand_minus_book_book_yaw_frame": rd(R.from_euler("z", -by).apply(t["ep"] - t["bp"])),
                    "hand_rel_book_quat_xyzw": rd((bq.inv() * hq).as_quat()), "hand_rel_book_euler_deg": rd((bq.inv() * hq).as_euler("xyz", degrees=True)),
                    "book_caddy_clearance_m": None if math.isnan(t["clr"]) else round(t["clr"], 4)}
        sel = sorted({kc, kc + 30, kpeak, k_above, first_ok, len(tr) - 1} - {None})
        path = [key(k) for k in range(max(kpeak, 0), len(tr), 6)] + [key(len(tr) - 1)]
        rec = {"demo": keys[di], "n_actions": len(acts), "replay_success": tr[-1]["ok"], "close_steps": close, "open_steps": opens,
               "releases": bool(opens), "first_env_success_step": first_ok,
               "site_world": {"pos": rd(sp), "R": [rd(r) for r in sR]},
               "key_poses": {str(k): key(k) for k in sel},
               "key_step_names": {"last_close": kc, "after_close_30": kc + 30, "peak_height": kpeak, "first_xy_inside_region": k_above,
                                  "first_env_success": first_ok, "last": len(tr) - 1},
               "descent_path_every_6_steps_from_peak": path,
               "min_book_caddy_clearance_after_peak_m": float(np.nanmin([t["clr"] for t in tr[kpeak:-1]])),
               "clearance_at_min_step": int(kpeak + np.nanargmin([t["clr"] for t in tr[kpeak:-1]]))}
        json.dump(rec, open(os.path.join(OUT, f"place_{keys[di]}.json"), "w"), indent=1)
        print("PE", keys[di], "success", tr[-1]["ok"], "releases", bool(opens), "close", kc, "peak", kpeak, "xy-in", k_above, "first_ok", first_ok,
              "n", len(acts), "min_clr", round(rec["min_book_caddy_clearance_after_peak_m"], 4), "at", rec["clearance_at_min_step"])
