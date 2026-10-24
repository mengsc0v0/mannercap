"""Dev check (read-only): is the planner's exact-0.0 mj_geomDistance a real touch or a CCD artifact?  Independent reference =
min distance from the hand geom's mesh vertices / box corners to the object's box geom (an UPPER bound of the true distance; misses
edge-face contact).  Usage: s7_zero_dist_validate.py TASK INIT DZ_CM"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
import synthesis_v2.approach_planner as ap
from s7_task_config import make_program, make_plan, make_session
from synthesis_v2.program_executor import produce_program_episode

T, I, DZ = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) / 100
ap.candidate_grid = lambda *a, **k: [(0, 0.0, DZ)]
LADDER = (0.05, 0.02, 0.01, 0.005, 0.002)
orig_dist = ap._dist
rows = []

def verts(m, d, g):
    R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]
    if m.geom_type[g] == 7:
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = m.mesh_vert[a:a + n]
    elif m.geom_type[g] == 6:
        s = m.geom_size[g]; V = np.array([[x * s[0], y * s[1], z * s[2]] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    else:
        return None
    return V @ R.T + c

def vdist(m, d, g, og):
    if m.geom_type[og] != 6: return None
    V = verts(m, d, g)
    if V is None: return None
    R = d.geom_xmat[og].reshape(3, 3); loc = (V - d.geom_xpos[og]) @ R
    ex = np.abs(loc) - m.geom_size[og]
    out = np.linalg.norm(np.maximum(ex, 0), axis=1)
    return float(out.min()) if (ex.max(axis=1) > 0).any() else -1.0     # -1: a vertex inside the box

def ladder(mj, m, data, g, og):
    ft = np.zeros(6)
    for dm in LADDER:
        r = float(mj.mj_geomDistance(m, data, g, og, dm, ft))
        if r != 0.0: return r
    return 0.0

def dist(mj, m, data, g, og, fromto):
    r = orig_dist(mj, m, data, g, og, fromto)
    v = vdist(m, data, g, og)
    rows.append((r, ladder(mj, m, data, g, og), v, int(m.geom_type[g])))
    return r
ap._dist = dist
prog = make_program(T); s = make_session(T, I, prog)
ep = produce_program_episode(prog, I, make_plan(T, prog, approach_check=True), keep_images=False, session=s)
A = np.array([(r, l, -9 if v is None else v, t) for r, l, v, t in rows])
print("queries", len(A), "status", ep.status)
z = A[A[:, 0] == 0.0]
print("planner==0.0:", len(z), " ladder==0.0 among them:", int((z[:, 1] == 0.0).sum()), " vertex-dist>=0.002 among them:", int((z[:, 2] >= 0.002).sum()),
      " vertex-dist in [0,0.002):", int(((z[:, 2] >= 0) & (z[:, 2] < 0.002)).sum()), " vertex inside box:", int((z[:, 2] == -1).sum()))
print("vertex-dist quantiles of planner==0.0 queries:", np.round(np.quantile(z[z[:, 2] >= 0, 2], [0, .25, .5, .75, 1]), 4).tolist() if len(z) else None)
nz = A[(A[:, 0] != 0.0) & (A[:, 2] >= 0)]
print("planner!=0.0:", len(nz), " |planner - vertex| median %.4f max %.4f; planner > vertex+0.002 (non-conservative): %d" % (np.median(np.abs(nz[:, 0] - nz[:, 2])), np.abs(nz[:, 0] - nz[:, 2]).max(), int((nz[:, 0] > nz[:, 2] + 0.002).sum())))
lz = A[(A[:, 1] != A[:, 0]) & (A[:, 2] >= 0)]
print("ladder differs from planner:", len(lz), " ladder above the vertex upper bound (+2mm; would be a violation):", int((A[:, 1][A[:, 2] >= 0] > A[:, 2][A[:, 2] >= 0] + 0.002).sum()),
      " planner above it:", int((A[:, 0][A[:, 2] >= 0] > A[:, 2][A[:, 2] >= 0] + 0.002).sum()))
bad = A[(A[:, 2] >= 0) & (A[:, 0] != 0.0)]
print("hand geom types among planner==0.0:", sorted(set(z[:, 3].astype(int).tolist())), "  raw zero rows (planner, ladder, vertex, type):", np.round(z, 4).tolist())
# Scope of the distance query: how many ladder answers are only a lower bound ("nothing closer than distmax", value == a rung)
# vs a returned distance, and how the returned distances compare with the vertex upper bound.
rung = np.isin(A[:, 1], LADDER)
print("ladder answers equal to a rung (lower bound only):", int(rung.sum()), "/", len(A),
      " below every rung (returned distance, incl. negative):", int((~rung).sum()),
      " exact 0.0:", int((A[:, 1] == 0.0).sum()))
d = A[(~rung) & (A[:, 2] >= 0) & (A[:, 1] != 0.0)]
print("returned distances vs vertex upper bound: n=%d, returned <= vertex+2mm: %d" % (len(d), int((d[:, 1] <= d[:, 2] + 0.002).sum())))
