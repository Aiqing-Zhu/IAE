"""Example 2: evaluate the interface-evolution operator.

Prediction is direct and time-conditioned -- no autoregression, no re-initialization:

    branch(z_geom(0), z_f, E[k]) --geometry head--> z_geom(t_k)
      --basis reconstruction--> signed-distance field --zero level set--> Gamma(t_k)

The number of connected components is therefore an output of the decoder, which is
what makes the merge measurable.

  python eval_evolution.py table            # 5-seed stratified d_H table (paper Table 2)
  python eval_evolution.py figure [ckpt]    # rollout figure (paper Figure 1)
"""

from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import torch
from contourpy import contour_generator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_evolution as T                                             # noqa: E402
from iae.encoders import basis_reconstruct                              # noqa: E402
from iae.paths import EX2_CACHE, EX2_CKPT, need                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIGDIR = HERE / "figures"

# Configurations of the paper table: (label, geometry head, truncation order r).
CONFIGS = [("nonlinear, r=24", "code", 24), ("nonlinear, r=16", "code", 16),
           ("nonlinear, r=12", "code", 12), ("linear,    r=24", "linear", 24)]
SEEDS = range(5)

FA = np.linspace(0.02, 0.98, 130)                    # decoding grid for the zero level set
FX, FY = np.meshgrid(FA, FA, indexing="ij")
FPTS = np.column_stack([FX.ravel(), FY.ravel()]).astype(np.float32)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def sdf_to_loops(sdf_fine, n_resample=130, min_perim=0.08):
    """Zero level set of a decoded field -> resampled closed loops (short ones dropped)."""
    out = []
    for seg in contour_generator(FX, FY, sdf_fine).lines(0.0):
        if len(seg) < 12:
            continue
        s = seg[:-1] if np.allclose(seg[0], seg[-1]) else seg
        if np.linalg.norm(np.diff(np.vstack([s, s[:1]]), axis=0), axis=1).sum() < min_perim:
            continue
        out.append(s[np.linspace(0, len(s) - 1, n_resample).astype(int)])
    return out


def hausdorff(a, b):
    """Two-sided Hausdorff distance between two sets of loops (NaN if either is empty)."""
    if not a or not b:
        return np.nan
    A, B = np.concatenate(a), np.concatenate(b)
    d = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
    return float(max(d.min(1).max(), d.min(0).max()))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(name):
    ck = torch.load(need(EX2_CKPT / name), map_location=T.DEV, weights_only=False)
    m = T.EvolutionOperator(ck["sg"][0].shape[1], ck["sf"][0].shape[1], ck["ntime"],
                            ck["order"], ck.get("geom_head", "code")).to(T.DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck


def predict_times(model, ck, z_geom0, zf0, ntimes):
    """From the INITIAL codes, decode the interface at each discrete time k = 0..ntimes-1."""
    sgm, sgs = ck["sg"]; sfm, sfs = ck["sf"]; szm, szs = ck["szg"]
    zg_t = torch.tensor(((z_geom0 - sgm) / sgs).astype(np.float32), device=T.DEV)
    zf_t = torch.tensor(((zf0 - sfm) / sfs).astype(np.float32), device=T.DEV)
    preds = []
    with torch.no_grad():
        for k in range(ntimes):
            bp = model.branch(zg_t, zf_t, torch.tensor([k], device=T.DEV))
            code = model.geom_code(bp)[0].cpu().numpy() * szs.ravel() + szm.ravel()
            sdf = basis_reconstruct(code, FPTS, basis=T.BASIS).reshape(FX.shape)
            preds.append(sdf_to_loops(sdf))
    return preds


def load_test():
    """Encoded test trajectories (cached by `train_evolution.py encode`)."""
    return pickle.load(open(need(EX2_CACHE / "test_enc.pkl"), "rb"))


# --------------------------------------------------------------------------- #
# Table: stratified 5-seed d_H
# --------------------------------------------------------------------------- #
def eval_ck(name, te):
    model, ck = load_model(name)
    allh, merge, nonmerge, offs = [], [], [], []
    for tr in te:
        preds = predict_times(model, ck, tr[0]["zgeom"], tr[0]["zf"], len(tr) - 1)
        hs = [h for h in (hausdorff(preds[k], tr[k + 1]["loops"]) for k in range(len(preds)))
              if not np.isnan(h)]
        allh += hs
        is_merge = tr[0]["ncomp"] > tr[-1]["ncomp"]
        (merge if is_merge else nonmerge).extend(hs)
        if is_merge:                                    # merge-frame offset, predicted vs reference
            rnc = [f["ncomp"] for f in tr]
            rmg = next((k for k in range(1, len(rnc)) if rnc[k] == 1), len(rnc))
            pnc = [len(p) for p in preds]
            pmg = next((k + 1 for k in range(len(pnc)) if pnc[k] == 1), len(rnc))
            offs.append(pmg - rmg)
    return (np.mean(allh), np.mean(merge) if merge else np.nan,
            np.mean(nonmerge) if nonmerge else np.nan, np.mean(offs) if offs else 0.0)


def table():
    te = load_test()
    nm = sum(1 for tr in te if tr[0]["ncomp"] > tr[-1]["ncomp"])
    print(f"test: {len(te)} trajectories ({nm} merge, {len(te)-nm} non-merge)\n", flush=True)
    print(f"{'configuration':18s} {'overall d_H':>17s} {'merge':>17s} {'non-merge':>17s} {'offset':>7s}")
    for label, head, order in CONFIGS:
        res = [eval_ck(n, te) for n in (T.ckpt_name(head, order, s) for s in SEEDS)
               if (EX2_CKPT / n).exists()]
        if not res:
            print(f"{label:18s}  (no checkpoints)"); continue
        R = np.array(res)
        cells = "  ".join(f"{R[:, i].mean():.4f} +- {R[:, i].std():.4f}" for i in range(3))
        print(f"{label:18s} {cells}  {R[:, 3].mean():>+7.2f}   [{len(res)} seeds]", flush=True)


# --------------------------------------------------------------------------- #
# Figure: rollout grid
# --------------------------------------------------------------------------- #
def figure(name, n_merge=4, n_single=1):
    """One row per test trajectory, one column per time; column 0 is the operator input."""
    model, ck = load_model(name)
    te = load_test()
    mi = [i for i, t in enumerate(te) if t[0]["ncomp"] > t[-1]["ncomp"]]      # two drops -> one
    si = [i for i, t in enumerate(te) if t[0]["ncomp"] == 1]                  # single drop
    sel = mi[:n_merge] + si[:n_single]
    ncol = max(len(te[i]) for i in sel)
    fig, axes = plt.subplots(len(sel), ncol, figsize=(2.4 * ncol, 2.5 * len(sel)), squeeze=False)
    for r, ti in enumerate(sel):
        traj = te[ti]
        preds = predict_times(model, ck, traj[0]["zgeom"], traj[0]["zf"], len(traj) - 1)
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(traj):
                ax.axis("off"); continue
            for lp in traj[c]["loops"]:                 # reference (column 0 = input, solid)
                p = np.vstack([lp, lp[:1]])
                ax.plot(p[:, 0], p[:, 1], "k-" if c == 0 else "k--", lw=1.3,
                        label=("initial input" if c == 0 else "reference")
                        if lp is traj[c]["loops"][0] else None)
            if c > 0:
                for lp in preds[c - 1]:
                    p = np.vstack([lp, lp[:1]])
                    ax.plot(p[:, 0], p[:, 1], "r-", lw=1.5,
                            label="predicted" if lp is preds[c - 1][0] else None)
            ax.set_aspect("equal"); ax.set_xlim(0.05, 0.95); ax.set_ylim(0.05, 0.95)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("initial (input)" if c == 0 else f"t={traj[c]['time']:.2f}", fontsize=15)
            if c == 0:
                lab = "merge" if traj[0]["ncomp"] > traj[-1]["ncomp"] else \
                      ("single" if traj[0]["ncomp"] == 1 else "far")
                ax.set_ylabel(f"traj {ti}\n[{lab}]", fontsize=14)
            if r == 0 and c == 1:
                ax.legend(fontsize=12, loc="upper right")
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    path = FIGDIR / "ex2_rollout.png"
    fig.savefig(path, dpi=125); plt.close(fig)
    print(f"saved {path}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    if cmd == "table":
        table()
    elif cmd == "figure":
        figure(sys.argv[2] if len(sys.argv) > 2 else T.ckpt_name("code", 24, 0))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
