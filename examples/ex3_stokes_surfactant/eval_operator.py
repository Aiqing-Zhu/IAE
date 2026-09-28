"""Example 3: the paper's Table 3 -- interface error and joint-state error.

Both metrics are DIRECT two-sided Hausdorff distances over the WHOLE state, with no
per-loop matching and no per-drop normalization:

    d_H   between the union of all predicted loops and the union of all reference
          loops, Euclidean, after mapping the physical box affinely to [0,1]^2
    d_gr  between the state graphs {(x, f(x))} under d((x,s),(y,t)) =
          max(||x - y||, |s - t|); x is box-normalized, the surfactant coordinate
          keeps its nondimensional scale

One value per (trajectory, time) frame, averaged per interaction regime, then
mean +- population std over the five seeds.  Loop-count agreement (predicted number
of drops == reference) is reported alongside.

  python eval_operator.py [checkpoint stem = surf_operator]
"""

from __future__ import annotations

import os
import pathlib
import pickle
import sys
from collections import defaultdict

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
from scipy.spatial.distance import cdist                                # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from iae.encoders import decode_interface, decode_surface_function      # noqa: E402
from iae.paths import EX3_CACHE, EX3_CKPT, need                         # noqa: E402
from train_operator import DEV, SurfOperator                            # noqa: E402

STEM = sys.argv[1] if len(sys.argv) > 1 else "surf_operator"
MODES = ["pass", "single", "row"]                 # shearing-past / isolated / co-translating
BOX_UNIT = (-7.0, 7.0, -5.0, 5.0)                 # physical box mapped to [0,1]^2
SEEDS = range(5)
DECODE_GRID = 240


def hd(A, B):
    """Two-sided Euclidean Hausdorff distance between point sets."""
    D = cdist(A, B); return max(D.min(1).max(), D.min(0).max())


def hd_graph(P, Q):
    """Two-sided Hausdorff distance under d((x,s),(y,t)) = max(||x-y||, |s-t|)."""
    D = np.maximum(cdist(P[:, :2], Q[:, :2]), np.abs(P[:, 2][:, None] - Q[:, 2][None, :]))
    return max(D.min(1).max(), D.min(0).max())


def load_model(ckf):
    ck = torch.load(need(EX3_CKPT / ckf), map_location=DEV, weights_only=False)
    order, ntime = ck["order"], ck["ntime"]
    m = SurfOperator(order * order, order * order, ntime, order).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    scal = {k: (ck[k][0].reshape(-1), ck[k][1]) for k in ("sZg", "sZgam", "sTg", "sTgam")}
    return m, ck, scal


def eval_ck(ckf, te, unit):
    m, ck, s = load_model(ckf)
    box = tuple(ck["box"])
    IF, GR = defaultdict(list), defaultdict(list)
    AVAIL, EMPTY = defaultdict(int), defaultdict(int)   # reference frames / empty predictions
    topo_ok = topo_tot = 0
    for tr in te:
        mode = tr[0].get("mode")
        zg0 = ((tr[0]["zgeom"] - s["sZg"][0]) / s["sZg"][1]).astype(np.float32)
        zgam0 = ((tr[0]["zgam"] - s["sZgam"][0]) / s["sZgam"][1]).astype(np.float32)
        for k in range(1, len(tr)):
            with torch.no_grad():
                pg, pgam = m(torch.tensor(zg0[None], device=DEV), torch.tensor(zgam0[None], device=DEV),
                             torch.tensor([k - 1], device=DEV))
            zg = pg.cpu().numpy()[0] * s["sTg"][1] + s["sTg"][0]
            zgam = pgam.cpu().numpy()[0] * s["sTgam"][1] + s["sTgam"][0]
            loops = [np.asarray(l, float) for l in decode_interface(zg, box, grid_n=DECODE_GRID)
                     if len(l) > 8]
            ref_loops = [np.asarray(l, float) for l in tr[k]["loops"]]
            ref_vals = [np.asarray(v, float) for v in tr[k]["fvals"]]
            AVAIL[mode] += 1
            topo_tot += 1; topo_ok += int(len(loops) == len(ref_loops))
            if not loops:
                EMPTY[mode] += 1
                continue
            Pp, Qp = unit(np.vstack(loops)), unit(np.vstack(ref_loops))
            IF[mode].append(hd(Pp, Qp))
            pf = np.concatenate([decode_surface_function(zgam, l, box) for l in loops])
            GR[mode].append(hd_graph(np.column_stack([Pp, pf]),
                                     np.column_stack([Qp, np.concatenate(ref_vals)])))
    overall = lambda D: np.mean(sum((D[x] for x in MODES), []))
    per = {mo: (np.mean(IF[mo]), np.mean(GR[mo]), len(IF[mo])) for mo in MODES}
    per["overall"] = (overall(IF), overall(GR), sum(len(IF[x]) for x in MODES))
    cnt = {mo: (AVAIL[mo], EMPTY[mo], len(IF[mo])) for mo in MODES}
    cnt["overall"] = (sum(AVAIL[x] for x in MODES), sum(EMPTY[x] for x in MODES), per["overall"][2])
    return per, topo_ok, topo_tot, cnt


def main():
    te = pickle.load(open(need(EX3_CACHE / "traj_test_enc.pkl"), "rb"))["trajs"]
    xlo, xhi, ylo, yhi = BOX_UNIT
    unit = lambda p: (np.asarray(p, float) - [xlo, ylo]) / [xhi - xlo, yhi - ylo]
    cols = MODES + ["overall"]

    print(f"{STEM}: direct two-sided Hausdorff, no per-loop matching, [0,1]^2 box\n")
    print(f"{'seed':>4} | " + " ".join(f"{mo}:d_H/d_gr" for mo in cols))
    rows, topo, cnts = [], [], []
    for sd in SEEDS:
        ckf = f"{STEM}-s{sd}.pt"
        if not (EX3_CKPT / ckf).exists():
            continue
        per, tok, ttot, cnt = eval_ck(ckf, te, unit)
        rows.append(per); topo.append((tok, ttot)); cnts.append(cnt)
        print(f"{sd:>4} | " + " ".join(f"{per[mo][0]:.4f}/{per[mo][1]:.4f}" for mo in cols)
              + f"   loop-count {tok}/{ttot}", flush=True)
    if not rows:
        raise SystemExit(f"no checkpoints {STEM}-s*.pt under {EX3_CKPT}")

    print(f"\n{len(rows)}-seed mean(std):")
    print(f"{'regime':>10} {'#frames':>8} {'interface d_H':>18} {'state d_gr':>18}")
    for mo in cols:
        I = np.array([r[mo][0] for r in rows]); G = np.array([r[mo][1] for r in rows])
        print(f"{mo:>10} {rows[0][mo][2]:>8} {I.mean():>10.4f}({I.std():.4f})  "
              f"{G.mean():>10.4f}({G.std():.4f})")
    print(f"loop-count agreement per seed: {[t[0] for t in topo]} / {topo[0][1]}")
    print("\nframe accounting (reference frames | empty predictions | denominator):")
    print(f"{'seed':>4} " + " ".join(f"{mo:>26}" for mo in cols))
    for sd, cnt in enumerate(cnts):
        print(f"{sd:>4} " + " ".join(f"{cnt[mo][0]:>8} |{cnt[mo][1]:>6} |{cnt[mo][2]:>8}" for mo in cols))


if __name__ == "__main__":
    main()
