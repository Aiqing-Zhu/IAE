"""Example 3: the paper's gallery figure.

One representative held-out trajectory per interaction regime (shearing-past,
isolated, co-translating), showing the reference (TRUE) above the prediction
(PRED) at t = 0, 2, 4, 6, 8, with the interfaces colored by the surfactant
concentration.  At t = 0 the prediction is the initial state, i.e. the operator's
input.

  python plot_gallery.py [checkpoint = surf_operator-s0.pt]   ->  figures/ex3_gallery.png
"""

from __future__ import annotations

import os
import pathlib
import pickle
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
from matplotlib.collections import LineCollection                       # noqa: E402
from scipy.spatial.distance import cdist                                # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_operator import load_model                                    # noqa: E402
from iae.encoders import decode_interface, decode_surface_function      # noqa: E402
from iae.paths import EX3_CACHE, need                                   # noqa: E402
from train_operator import DEV                                          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CKPT = sys.argv[1] if len(sys.argv) > 1 else "surf_operator-s0.pt"
TIMES = [0, 2, 4, 6, 8]
DECODE_GRID = 260
REGIMES = [("shearing-past", "pass"), ("isolated", "single"), ("co-translating row", "row")]


def min_gap(tr):
    """Smallest distance between the two drops over a trajectory."""
    g = np.inf
    for fr in tr:
        L = [np.asarray(l, float) for l in fr["loops"]]
        if len(L) == 2:
            g = min(g, cdist(L[0], L[1]).min())
    return g


def max_elongation(tr):
    def el(lp):
        p = np.asarray(lp, float); p = p - p.mean(0); w = np.linalg.eigvalsh(np.cov(p.T))
        return (max(w[1], 1e-9) / max(w[0], 1e-9)) ** 0.5
    return max(el(l) for fr in tr for l in fr["loops"])


def main():
    m, ck, s = load_model(CKPT)
    box = tuple(ck["box"])
    te = pickle.load(open(need(EX3_CACHE / "traj_test_enc.pkl"), "rb"))["trajs"]
    full = [tr for tr in te if len(tr) == ck["ntime"] + 1]

    def predict(tr, t):
        zg0 = ((tr[0]["zgeom"] - s["sZg"][0]) / s["sZg"][1]).astype(np.float32)
        zgam0 = ((tr[0]["zgam"] - s["sZgam"][0]) / s["sZgam"][1]).astype(np.float32)
        with torch.no_grad():
            pg, pgam = m(torch.tensor(zg0[None], device=DEV), torch.tensor(zgam0[None], device=DEV),
                         torch.tensor([t - 1], device=DEV))
        zg = pg.cpu().numpy()[0] * s["sTg"][1] + s["sTg"][0]
        zgam = pgam.cpu().numpy()[0] * s["sTgam"][1] + s["sTgam"][0]
        loops = [np.asarray(l, float) for l in decode_interface(zg, box, grid_n=DECODE_GRID)
                 if len(l) > 8]
        return loops, [decode_surface_function(zgam, l, box) for l in loops]

    def topology_ok(tr):                       # predicted drop count matches at every shown time
        return all(len(predict(tr, t)[0]) == len(tr[t]["loops"]) for t in TIMES if t > 0)

    def pick(mode):
        c = [tr for tr in full if tr[0].get("mode") == mode]
        if not c:
            return None
        if mode == "pass":                     # the closest squeeze-and-cross
            return min(c, key=min_gap)
        pool = sorted([tr for tr in c if topology_ok(tr)] or c, key=max_elongation)
        return pool[int(0.6 * (len(pool) - 1))]        # representative, not extreme, deformation

    chosen = [(lab, pick(mode)) for lab, mode in REGIMES]
    chosen = [(lab, tr) for lab, tr in chosen if tr is not None]
    print("trajectories:", [(lab, int(tr[0].get("traj", -1))) for lab, tr in chosen], flush=True)

    allf = [np.concatenate(fr["fvals"]) for _, tr in chosen for fr in tr]
    norm = plt.Normalize(min(a.min() for a in allf), max(a.max() for a in allf))
    print(f"surfactant colorbar range: [{norm.vmin:.4f}, {norm.vmax:.4f}]", flush=True)

    def draw(ax, loops, fvals):
        for lp, fv in zip(loops, fvals):
            lp = np.asarray(lp); seg = np.vstack([lp, lp[:1]]); P = seg.reshape(-1, 1, 2)
            lc = LineCollection(np.concatenate([P[:-1], P[1:]], 1), cmap="viridis", norm=norm)
            fc = np.append(fv, fv[0]); lc.set_array(0.5 * (fc[:-1] + fc[1:])); lc.set_linewidth(2.0)
            ax.add_collection(lc)
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

    nrow, ncol = 2 * len(chosen), len(TIMES)
    cellw = 2.15; cellh = cellw * (box[3] - box[2]) / (box[1] - box[0])   # true domain aspect
    fig, axes = plt.subplots(nrow, ncol, figsize=(cellw * ncol, cellh * nrow), squeeze=False)
    for r, (label, tr) in enumerate(chosen):
        for c, t in enumerate(TIMES):
            axt, axp = axes[2 * r, c], axes[2 * r + 1, c]
            draw(axt, tr[t]["loops"], tr[t]["fvals"])                     # TRUE
            draw(axp, *((tr[0]["loops"], tr[0]["fvals"]) if t == 0 else predict(tr, t)))
            if r == 0:
                axt.set_title(f"$t={t}$", fontsize=17)
            if c == 0:
                axt.set_ylabel("TRUE", fontsize=15); axp.set_ylabel("PRED", fontsize=15)

    fig.subplots_adjust(left=0.035, right=0.915, top=0.955, bottom=0.01, wspace=0.05, hspace=0.05)
    cax = fig.add_axes([0.93, 0.12, 0.013, 0.76])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=norm), cax=cax)
    cb.set_label("surfactant $f$", fontsize=15); cb.ax.tick_params(labelsize=12)
    (HERE / "figures").mkdir(exist_ok=True)
    out = HERE / "figures" / "ex3_gallery.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out} | regimes: {[lab for lab, _ in chosen]}")


if __name__ == "__main__":
    main()
