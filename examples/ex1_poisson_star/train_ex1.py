"""Example 1: train and evaluate one (encoder, normalization, seed) configuration.

A four-branch MIONet  (geometry, k, f, g)  x  trunk(x)  is trained to predict the
Poisson solution u on variable star / notched domains.  The four encoders differ
only in how the geometry -- and, for IAE, the boundary datum g -- is coded:

    iae_cosine     IAE, cosine (DCT) basis          z_geom = SDF code, z_g = McShane code
    iae_legendre   IAE, tensor-Legendre basis       "
    mfe2d          2D-MFE baseline                  z_geom = domain moments of 1_Omega
    mfe1d          1D-MFE baseline                  z_geom = boundary moment of 1

k and f are Legendre domain moments in every case.  Reports the all-node relative
L2 error on the test set, split into all / notched / smooth domains, at the FINAL
checkpoint (no checkpoint selection on the test set); the best-checkpoint value is
printed in brackets for reference only.

  python train_ex1.py [encoder] [norm: none|perdim|lowfreq] [seed] [iters=120000]
"""
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iae.models import MIONet                                   # noqa: E402
from iae.paths import EX1_CACHE, need                           # noqa: E402

ENCODERS = ("iae_cosine", "iae_legendre", "mfe2d", "mfe1d")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "iae_cosine"
NORM = sys.argv[2] if len(sys.argv) > 2 else "lowfreq"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 120000
ORDER, NUM_LOC, BATCH, RHO = 12, 300, 64, 0.5       # RHO = low-frequency weight strength
if METHOD not in ENCODERS:
    raise SystemExit(f"unknown encoder {METHOD!r}; choose from {ENCODERS}")
torch.manual_seed(SEED); np.random.seed(SEED)


def normalize(a_tr, a_te, order, mode, rho):
    """Normalize a code block. 'none' = raw; 'perdim' = per-coefficient standardization;
    'lowfreq' = global std, then weight coefficient (i,j) by 1/(1+rho(i^2+j^2))."""
    if mode == "none":
        return a_tr.astype(np.float32), a_te.astype(np.float32)
    o2 = order * order; m = a_tr.mean(0, keepdims=True); tr0, te0 = a_tr - m, a_te - m
    if mode == "perdim":
        sd = a_tr.std(0, keepdims=True) + 1e-8
        return (tr0 / sd).astype(np.float32), (te0 / sd).astype(np.float32)
    if mode != "lowfreq":
        raise SystemExit(f"unknown normalization {mode!r}")
    g = float(tr0.std()) + 1e-8; tr1, te1 = tr0 / g, te0 / g
    ii, jj = np.meshgrid(np.arange(order), np.arange(order), indexing="ij")
    w = 1.0 / (1.0 + rho * (ii.ravel() ** 2 + jj.ravel() ** 2)); w = w / w.mean()
    wf = np.tile(w, a_tr.shape[1] // o2).astype(np.float32)
    return (tr1 * wf).astype(np.float32), (te1 * wf).astype(np.float32)


def load_codes():
    """(train, test) code blocks (z_geom, z_k, z_f, z_g) for the chosen encoder."""
    ctr = np.load(need(EX1_CACHE / "mfe_train.npz"))
    cte = np.load(need(EX1_CACHE / "mfe_test.npz"), allow_pickle=True)
    if METHOD.startswith("mfe"):
        geom = "mfe_zgeom1d" if METHOD == "mfe1d" else "mfe_zgeom"
        tr = (ctr[geom], ctr["mfe_zk"], ctr["mfe_zf"], ctr["mfe_zg"])
        te = (cte[geom], cte["mfe_zk"], cte["mfe_zf"], cte["mfe_zg"])
    else:
        basis = METHOD.split("_")[1]
        dtr = np.load(need(EX1_CACHE / f"iae_train_{basis}.npz"))
        dte = np.load(need(EX1_CACHE / f"iae_test_{basis}.npz"))
        tr = tuple(dtr[k] for k in ("zgeom", "zk", "zf", "zg"))
        te = tuple(dte[k] for k in ("zgeom", "zk", "zf", "zg"))
    return ctr, cte, tr, te


ctr, cte, tr_codes, te_codes = load_codes()
notch_te = cte["notch"].astype(bool)
print(f"{len(ctr['pos'])} train, {len(notch_te)} test | encoder={METHOD} norm={NORM} "
      f"seed={SEED} iters={ITERS}", file=sys.stderr, flush=True)

blocks = [normalize(a, b, ORDER, NORM, RHO) for a, b in zip(tr_codes, te_codes)]
T4 = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEV)
Ztr = [T4(x[0]) for x in blocks]; Zte = [T4(x[1]) for x in blocks]
pos, yv = T4(ctr["pos"]), T4(ctr["yv"])
pts_te = [T4(p[None]) for p in cte["points"]]; u_te = [T4(u) for u in cte["u"]]

P, H, D = 128, 256, 3
sizes = [[c.shape[1]] + [H] * D + [P] for c in tr_codes] + [[2] + [H] * D + [P]]
model = MIONet(sizes, "relu", bias=True).to(DEV)
opt = torch.optim.Adam(model.parameters(), 1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
mse = torch.nn.MSELoss()


def evaluate():
    """Per-sample all-node relative L2 error on the test set."""
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(len(u_te)):
            pred = model([z[i:i + 1] for z in Zte] + [pts_te[i]])[0]
            errs.append((torch.norm(pred - u_te[i]) / torch.norm(u_te[i])).item())
    model.train(); return np.array(errs)


n = len(pos); t = time.time(); best_all = 1e9; best = final = None
EVAL_EVERY = max(1, ITERS // 4)
for it in range(1, ITERS + 1):
    idx = torch.randint(0, n, (BATCH,), device=DEV)
    loss = mse(model([z[idx] for z in Ztr] + [pos[idx]]), yv[idx])
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it % EVAL_EVERY == 0:
        e = evaluate()
        final = (e.mean(), e[notch_te].mean(), e[~notch_te].mean())
        if e.mean() < best_all:
            best_all = e.mean(); best = final
        print(f"  it {it}: all {e.mean():.4f} ({time.time()-t:.0f}s)", file=sys.stderr, flush=True)

print(f">>> {METHOD:13s} {NORM:8s} seed={SEED}: all {final[0]:.4f} | notched {final[1]:.4f} "
      f"| smooth {final[2]:.4f}  [best-ckpt {best[0]:.4f}]", flush=True)
print(f"RESULT,{METHOD},{NORM},{SEED},{final[0]:.6f},{final[1]:.6f},{final[2]:.6f}", flush=True)
