"""GINO baseline for Example 1 (paper Table 1, last row).

GINO (Geometry-Informed Neural Operator, Li et al., NeurIPS 2023) is trained and
evaluated on the SAME data split as the MIONet rows, with the same five-seed
protocol, so the two are directly comparable as numbers even though the
architectures and the training budgets are not matched.

Input construction (unchanged from the reference implementation's usage): each
sample becomes a point cloud of the in-domain nodes of a 96 x 96 uniform grid plus
the boundary vertices, carrying four channels

    [k, f, 0, sdf]   on interior grid points        [k, f, g, 0]   on boundary points

where `sdf` is the signed distance to dOmega -- the same `signed_distance_from_edges`
the IAE uses, the difference being that GINO consumes point values while the IAE
consumes the projection coefficients.  The model queries the mesh nodes, so the
reported relative L2 error is computed on exactly the same nodes as for MIONet.

Reproducibility: the raw files are read in the fixed order raw_train.npz ->
raw_train_b.npz -> raw_test.npz (the order `encode_mfe.py` uses), so sample i is the
same domain for both methods; their sha256 sums, the seeds, the full convergence
history and the weights are stored with every run.

  python gino_baseline.py build-cache            # one-off: precompute the point clouds
  python gino_baseline.py verify-cache           # bitwise check of the cache against the raw data
  python gino_baseline.py train <seed> [iters=120000]
  python gino_baseline.py summarize [iters=120000] [dir=results]   # `paper_runs` = our runs

Needs the `neuraloperator` package (see baselines/gino/README.md).  If it is not
installed, set NEURALOP_PATH to a checkout of the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
if os.environ.get("NEURALOP_PATH"):
    sys.path.insert(0, os.environ["NEURALOP_PATH"])

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.tri as mtri                                           # noqa: E402
import torch                                                            # noqa: E402

from iae.encoders import signed_distance_from_edges                     # noqa: E402
from iae.paths import CACHE_ROOT, EX1_DATA, need, out_dir               # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CACHE_DIR = CACHE_ROOT / "gino"
CACHE = CACHE_DIR / "gino_inputs.pkl"
RESULTS = HERE / "results"
RAW_FILES = ("raw_train.npz", "raw_train_b.npz", "raw_test.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

GRID_RES = 96
_ga = np.linspace(0, 1, GRID_RES)
_GX, _GY = np.meshgrid(_ga, _ga)
GRID = np.column_stack([_GX.ravel(), _GY.ravel()])


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def sha256_file(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def data_checksums():
    return {f: sha256_file(need(EX1_DATA / f)) for f in RAW_FILES}


def load1(fp):
    d = np.load(fp); num = int(d["num"]); pp, tp, bp = d["point_ptr"], d["tri_ptr"], d["bedge_ptr"]
    P, T, BE, K, F, G, U = d["points"], d["triangles"], d["bedges"], d["k"], d["f"], d["g"], d["u"]
    out = []
    for i in range(num):
        a, b = pp[i], pp[i + 1]
        out.append(dict(points=P[a:b].astype(np.float64), triangles=T[tp[i]:tp[i + 1]] - a,
                        bedges=BE[bp[i]:bp[i + 1]] - a, k=K[a:b].astype(np.float64),
                        f=F[a:b].astype(np.float64), g=G[a:b].astype(np.float64),
                        u=U[a:b].astype(np.float64)))
    return out, d["notch"]


def load(split):
    files = ["raw_train.npz", "raw_train_b.npz"] if split == "train" else ["raw_test.npz"]
    O, N = [], []
    for fp in files:
        o, n = load1(need(EX1_DATA / fp)); O += o; N.append(n)
    return O, np.concatenate(N)


def build_input(s):
    """One sample -> (point cloud [N,2], channels [N,4])."""
    pts, tri, be = s["points"], s["triangles"], s["bedges"]
    sdf = signed_distance_from_edges(GRID, pts, be); inside = sdf < -0.006
    gpts, sdf_in = GRID[inside], sdf[inside]
    trian = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
    itp = lambda v: np.asarray(mtri.LinearTriInterpolator(trian, v)(gpts[:, 0], gpts[:, 1]).filled(0.0))
    k_in, f_in = itp(s["k"]), itp(s["f"])
    bn = np.unique(be); bpts = pts[bn]
    allp = np.vstack([gpts, bpts]).astype(np.float32); ng = len(gpts)
    x = np.zeros((len(allp), 4), np.float32)
    x[:ng, 0], x[:ng, 1], x[:ng, 3] = k_in, f_in, sdf_in
    x[ng:, 0], x[ng:, 1], x[ng:, 2] = s["k"][bn], s["f"][bn], s["g"][bn]
    return allp, x


def build_cache():
    out_dir(CACHE_DIR)
    t0 = time.time()
    prep = lambda S: [(*build_input(s), s["points"].astype(np.float32), s["u"].astype(np.float32))
                      for s in S]
    tr, _ = load("train"); te, _ = load("test")
    print(f"building point clouds for {len(tr)} train + {len(te)} test samples ...", flush=True)
    pickle.dump({"tr": prep(tr), "te": prep(te)}, open(CACHE, "wb"), protocol=4)
    print(f"wrote {CACHE} ({CACHE.stat().st_size/1e9:.1f} GB, {time.time()-t0:.0f}s)")


def verify_cache(k_train=15, k_test=5, seed=0):
    """Spot-check bitwise that the cached inputs were built from THIS raw data."""
    print(f"raw dir: {EX1_DATA}")
    for k, v in data_checksums().items():
        print(f"  {k}: sha256 {v}")
    tr, _ = load("train"); te, _ = load("test")
    data = pickle.load(open(need(CACHE), "rb"))
    print(f"cache {CACHE}: {len(data['tr'])} train / {len(data['te'])} test entries; "
          f"raw: {len(tr)} train / {len(te)} test")
    if len(data["tr"]) != len(tr) or len(data["te"]) != len(te):
        print("  LENGTH MISMATCH -> the cache does not correspond to this data"); return 1
    rng = np.random.default_rng(seed); bad = 0
    for tag, S, C, kk in (("train", tr, data["tr"], k_train), ("test", te, data["te"], k_test)):
        idx = rng.choice(len(S), size=min(kk, len(S)), replace=False)
        for i in idx:
            gp, x = build_input(S[int(i)])
            cgp, cx, cp, cu = C[int(i)]
            if not (np.array_equal(gp, cgp) and np.array_equal(x, cx)
                    and np.array_equal(S[int(i)]["points"].astype(np.float32), cp)
                    and np.array_equal(S[int(i)]["u"].astype(np.float32), cu)):
                bad += 1; print(f"  MISMATCH {tag}[{i}]")
        print(f"  {tag}: {len(idx)} random samples checked bitwise")
    print(f"\ncache verification: {'PASS' if bad == 0 else f'FAIL ({bad} mismatches)'}")
    return 0 if bad == 0 else 1


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def import_neuralop():
    """Imported lazily: only training needs the library (see this directory's README)."""
    try:
        import neuralop
        from neuralop.models import GINO
    except ImportError as exc:                                          # noqa: BLE001
        raise SystemExit(f"{exc}\ninstall it with `pip install neuraloperator==2.0.0`, or set "
                         "NEURALOP_PATH to a checkout of the repository") from exc
    return neuralop, GINO


def make_model(GINO):
    return GINO(in_channels=4, out_channels=1, gno_coord_dim=2,
                in_gno_radius=0.12, out_gno_radius=0.12,
                in_gno_channel_mlp_hidden_layers=[64, 64],
                out_gno_channel_mlp_hidden_layers=[64, 64],
                fno_in_channels=4, fno_n_modes=(16, 16), fno_hidden_channels=48,
                fno_n_layers=4, gno_use_open3d=False).to(DEV)


def train(seed, iters):
    neuralop, GINO = import_neuralop()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    out_dir(RESULTS)
    t_all = time.time()
    cks = data_checksums()
    tr, _ = load("train"); te, te_notch = load("test")
    data = pickle.load(open(need(CACHE), "rb"))
    assert len(data["tr"]) == len(tr) and len(data["te"]) == len(te), \
        "cache / raw length mismatch -- rebuild with `build-cache`"

    togpu = lambda L: ([torch.tensor(gp, device=DEV) for gp, x, p, u in L],
                       [torch.tensor(x, device=DEV) for gp, x, p, u in L],
                       [torch.tensor(p[None], device=DEV) for gp, x, p, u in L],
                       [torch.tensor(u, device=DEV) for gp, x, p, u in L])
    G_tr, X_tr, OQ_tr, Y_tr = togpu(data["tr"])
    G_te, X_te, OQ_te, Y_te = togpu(data["te"])
    print(f"seed {seed}: {len(tr)} train {len(te)} test | torch {torch.__version__} "
          f"neuralop {neuralop.__version__}", flush=True)

    allx = torch.cat(X_tr, 0); xm, xs = allx.mean(0), allx.std(0) + 1e-8
    ally = torch.cat(Y_tr, 0); ym, ys = ally.mean().item(), ally.std().item() + 1e-8
    X_trn = [(x - xm) / xs for x in X_tr]; X_ten = [(x - xm) / xs for x in X_te]
    latent = torch.stack(torch.meshgrid(torch.linspace(0, 1, 24, device=DEV),
                                        torch.linspace(0, 1, 24, device=DEV), indexing="xy"), -1)
    model = make_model(GINO)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"GINO {nparam/1e3:.0f}k params, iters={iters}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    mse = torch.nn.MSELoss()

    def evaluate():
        model.eval(); errs = np.zeros(len(te))
        with torch.no_grad():
            for i in range(len(te)):
                pred = model(input_geom=G_te[i], latent_queries=latent, output_queries=OQ_te[i],
                             x=X_ten[i].unsqueeze(0)).squeeze(0).squeeze(-1) * ys + ym
                errs[i] = (torch.norm(pred - Y_te[i]) / torch.norm(Y_te[i])).item()
        model.train(); return errs

    # One geometry per forward pass, so one sample per update: the reference
    # implementation does the same (it squeezes the batch dimension of input_geom).
    rng = np.random.default_rng(seed)
    eval_every = max(2000, iters // 24)
    n = len(tr); t0 = time.time(); nm = te_notch > 0.5; history = []
    stem = RESULTS / f"gino-s{seed}-it{iters}"
    for it in range(1, iters + 1):
        i = int(rng.integers(n)); yn = ((Y_tr[i] - ym) / ys).unsqueeze(-1)
        pred = model(input_geom=G_tr[i], latent_queries=latent, output_queries=OQ_tr[i],
                     x=X_trn[i].unsqueeze(0)).squeeze(0)
        loss = mse(pred, yn); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % eval_every == 0:
            e = evaluate()
            history.append(dict(iter=it, all=float(e.mean()), notch=float(e[nm].mean()),
                                smooth=float(e[~nm].mean()), seconds=round(time.time() - t0, 1)))
            print(f"  iter {it}: all {e.mean():.4f} notched {e[nm].mean():.4f} "
                  f"smooth {e[~nm].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
            torch.save(dict(model=model.state_dict(), xm=xm.cpu(), xs=xs.cpu(), ym=ym, ys=ys,
                            seed=seed, iter=it, history=history),
                       stem.with_suffix(".partial.pt"))      # a multi-day run should survive a crash

    e = evaluate()
    res = dict(seed=seed, iters=iters, params=int(nparam), all=float(e.mean()),
               notch=float(e[nm].mean()), smooth=float(e[~nm].mean()),
               n_train=len(tr), n_test=len(te), data_dir=str(EX1_DATA), data_sha256=cks,
               torch=torch.__version__, neuralop=neuralop.__version__,
               python=sys.version.split()[0], seconds=round(time.time() - t_all, 1),
               eval_every=eval_every, history=history)
    torch.save(dict(model=model.state_dict(), xm=xm.cpu(), xs=xs.cpu(), ym=ym, ys=ys, **res),
               stem.with_suffix(".pt"))
    json.dump(res, open(stem.with_suffix(".json"), "w"), indent=1)
    print(f">>> GINO seed {seed}: all {res['all']:.4f} | notched {res['notch']:.4f} | "
          f"smooth {res['smooth']:.4f}  ({nparam/1e3:.0f}k params, {iters} iters, "
          f"{res['seconds']:.0f}s) -> {stem.name}.pt", flush=True)


def summarize(iters, where=None):
    """Aggregate the per-seed JSON results into the Table-1 GINO row.

    Also prints the convergence statistic the paper's appendix quotes -- the absolute change
    in overall test error over the final evaluation interval -- so that claim can be checked
    directly against the retained run histories."""
    d = pathlib.Path(where) if where else RESULTS
    if not d.is_absolute():
        d = HERE / d
    files = sorted(d.glob(f"gino-s*-it{iters}.json"))
    if not files:
        raise SystemExit(f"no results gino-s*-it{iters}.json under {d}")
    rows = [json.load(open(f)) for f in files]

    A = np.array([[r["all"], r["notch"], r["smooth"]] for r in rows])
    m, sd = A.mean(0), A.std(0)                                # population std, as in the paper
    print(f"{len(rows)} seeds, {iters} updates:")
    for i, name in enumerate(("overall", "notched", "smooth")):
        print(f"  {name:8s} {m[i]:.4f} +- {sd[i]:.4f}")
    print(f"  total compute {sum(r['seconds'] for r in rows)/3600:.1f} GPU-hours")

    ev = rows[0]["eval_every"]
    print(f"\nconvergence (evaluated every {ev} updates):")
    print(f"{'seed':>4} {'all':>8} {'notched':>8} {'smooth':>8}   {'last three evaluations':>26}"
          f"   {'|change| over':>14}")
    print(f"{'':>4} {'':>8} {'':>8} {'':>8}   {'':>26}   {'final interval':>14}")
    final, window = [], []
    for r in rows:
        h = [x["all"] for x in r["history"]]
        final.append(abs(h[-1] - h[-2]))
        window.append(max(abs(h[i + 1] - h[i]) for i in (-3, -2)))
        tail = ", ".join(f"{v:.4f}" for v in h[-3:])
        print(f"{r['seed']:>4} {r['all']:8.4f} {r['notch']:8.4f} {r['smooth']:8.4f}   {tail:>26}"
              f"   {final[-1]:14.1e}")
    lo, hi = rows[0]["history"][-2]["iter"], rows[0]["history"][-1]["iter"]
    print(f"  max over seeds, between updates {lo} and {hi} : {max(final):.1e}")
    print(f"  max over seeds, over the last three evaluations   : {max(window):.1e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build-cache":
        build_cache()
    elif cmd == "verify-cache":
        raise SystemExit(verify_cache())
    elif cmd == "summarize":
        summarize(int(sys.argv[2]) if len(sys.argv) > 2 else 120000,
                  sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "train" and len(sys.argv) > 2:
        train(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 120000)
    elif cmd == "train":
        raise SystemExit("train needs a seed: python gino_baseline.py train <seed> [iters]")
    else:
        raise SystemExit(__doc__)
