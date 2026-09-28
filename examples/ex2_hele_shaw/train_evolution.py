"""Example 2: the time-conditioned interface-evolution operator for Hele-Shaw flow.

The operator maps the INITIAL state and a discrete time directly to the interface
at that time -- there is no autoregression and no re-initialization, so a merge is
predicted in one shot from t = 0:

    input   z_geom(0)   IAE code of Gamma(0)  (signed-distance field -> basis)
            z_f         code of the source s
            k           discrete time index, t_k = 0.56 k
    branch  b_g(z_geom(0)) * b_f(z_f) * E[k]        E = learned per-time embedding
    heads   geometry -> z_geom(t_k) DIRECTLY (the network output IS the spectral
              code; Gamma(t_k) is the zero level set of the field reconstructed
              from it)
            field    -> p(., t_k) on a spatial trunk (auxiliary supervision)

Two geometry heads are compared in the paper: 'code' (an MLP with hidden layers)
and 'linear' (a single linear read-out from the branch product).  The geometry
loss is a low-frequency-weighted coefficient MSE; the truncation order r is swept
by cutting the stored order-24 code down to r x r, which is exact for an
orthonormal basis.

  python train_evolution.py encode                    # once: encode + cache pairs (needs the meshes)
  python train_evolution.py reencode [grid] [order]   # rebuild the cache from interfaces.pkl alone
  python train_evolution.py [iters] [head] [order] [seed]
"""

from __future__ import annotations

import pathlib
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iae.encoders import gpu                                            # noqa: E402
from iae.encoders import basis_1d, union_sdf                            # noqa: E402
from iae.models import FNN                                              # noqa: E402
from iae.paths import EX2_CACHE, EX2_CKPT, EX2_DATA, need, out_dir      # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GEOM_GRID, GEOM_ORDER, MFE_MODE = 48, 24, 12  # encoder grid / stored code order / source moment order
FREQ_P = 0.02                                 # low-frequency weight of the code loss
W_FIELD = 0.3                                 # weight of the auxiliary pressure loss
BATCH = 32
_A = (np.arange(GEOM_GRID) + 0.5) / GEOM_GRID

# Cosine only: the geometry head predicts a code that is reconstructed back into a
# field, and the grid projection below inverts exactly for the cosine basis (it is a
# DCT-II) but not for Legendre -- see iae/encoders/extension.py.  The input z_geom(0)
# and the code TARGET must use the same projection, so the encoder is forced to grid
# projection ('dct').
BASIS = "cosine"
ENC_KW = dict(geom_basis=BASIS, geom_order=GEOM_ORDER, geom_mode="dct",
              geom_grid=GEOM_GRID, mfe_mode=MFE_MODE)
_BB = basis_1d(_A, GEOM_ORDER, BASIS)


def sdf_to_code(sdf):                        # [.,grid,grid] -> [.,order^2]
    return np.einsum("...xy,xi,yj->...ij", sdf, _BB, _BB).reshape(*sdf.shape[:-2], -1) / GEOM_GRID ** 2


def grid_xy():
    gx, gy = np.meshgrid(_A, _A, indexing="ij")
    return gx, gy


def freq_weights(order, freq_p):
    """Low-frequency weights 1/(1 + p(i^2 + j^2)) of the coefficient loss, mean-normalized."""
    ii, jj = np.meshgrid(np.arange(order), np.arange(order), indexing="ij")
    w = 1.0 / (1.0 + freq_p * (ii.ravel() ** 2 + jj.ravel() ** 2))
    return (w / w.mean()).astype(np.float32)


# --------------------------------------------------------------------------- #
# encode
# --------------------------------------------------------------------------- #
def encode_frames(trajs):
    """Attach the IAE codes and the SDF target grid to every frame (in place)."""
    flat = [f for t in trajs for f in t]
    codes = gpu.encode_batch(flat, encoder="iae", device=DEV, chunk=8, **ENC_KW)
    gx, gy = grid_xy()
    for i, f in enumerate(flat):
        f["zgeom"] = np.asarray(codes["zgeom"][i]); f["zf"] = np.asarray(codes["zf"][i])
        f["sdf24"] = union_sdf(f["loops"], gx, gy)
    return trajs


def build_pairs(trajs, n_field=300, rng=None):
    """Time-conditioned samples: (frame-0 codes, time index k) -> frame-k state."""
    rng = rng or np.random.default_rng(0)
    P = []
    for t in trajs:
        z0g, z0f = t[0]["zgeom"], t[0]["zf"]
        for k in range(1, len(t)):
            b = t[k]
            idx = rng.choice(len(b["points"]), n_field, replace=len(b["points"]) < n_field)
            P.append(dict(zg=z0g, zf=z0f, tk=k - 1, sdf_k=b["sdf24"].ravel(),
                          fpos=b["points"][idx].astype(np.float32), fval=b["u"][idx].astype(np.float32)))
    return P


def cache_encode():
    """Encode every frame and build the training pairs ONCE, then cache to disk."""
    out = out_dir(EX2_CACHE)
    tr = pickle.load(open(need(EX2_DATA / "traj_train.pkl"), "rb"))
    te = pickle.load(open(need(EX2_DATA / "traj_test.pkl"), "rb"))
    encode_frames(tr); encode_frames(te)
    P = build_pairs(tr, rng=np.random.default_rng(0))
    np.savez(out / "train_pairs.npz",
             zg=np.array([p["zg"] for p in P], np.float32), zf=np.array([p["zf"] for p in P], np.float32),
             tk=np.array([p["tk"] for p in P], np.int64),
             sdf_k=np.array([p["sdf_k"] for p in P], np.float32),
             fpos=np.array([p["fpos"] for p in P]), fval=np.array([p["fval"] for p in P]))
    te_min = [[dict(zgeom=f["zgeom"], zf=f["zf"], loops=f["loops"], time=f["time"], ncomp=f["ncomp"])
               for f in t] for t in te]
    pickle.dump(te_min, open(out / "test_enc.pkl", "wb"))
    print(f"cached {len(P)} train pairs + {len(te)} test trajectories "
          f"(grid {GEOM_GRID}, order {GEOM_ORDER}, {BASIS} basis) -> {out}", flush=True)


def reencode(grid=GEOM_GRID, order=GEOM_ORDER):
    """Rebuild the cache at a different projection grid / code order, from the distributed
    `data/ex2_hele_shaw/interfaces.pkl` alone -- no finite-element mesh needed.

    z_geom is a function of the interface only, so it is recomputed from the polylines (this
    reproduces the mesh-based encoder to ~3e-9).  The source code z_f and the auxiliary
    pressure samples do not depend on the geometry discretization and are carried over from
    the existing cache unchanged.  Frame order is identical, so the two line up row by row."""
    from iae.encoders import encode_interface
    global GEOM_GRID, GEOM_ORDER, _A, _BB
    out = out_dir(EX2_CACHE)
    iface = pickle.load(open(need(EX2_DATA / "interfaces.pkl"), "rb"))
    old = np.load(need(EX2_CACHE / "train_pairs.npz"))

    GEOM_GRID, GEOM_ORDER = grid, order                       # rebind the module-level grid
    _A = (np.arange(grid) + 0.5) / grid
    _BB = basis_1d(_A, order, BASIS)
    gx, gy = grid_xy()
    box = (0.0, 1.0, 0.0, 1.0)
    enc = lambda loops: encode_interface(loops, box, grid_n=grid, order=order, basis=BASIS)

    zg, sdf_k = [], []
    for t in iface["train"]:
        z0 = enc(t[0]["loops"]).astype(np.float32)
        for k in range(1, len(t)):
            zg.append(z0); sdf_k.append(union_sdf(t[k]["loops"], gx, gy).ravel().astype(np.float32))
    zg = np.array(zg, np.float32); sdf_k = np.array(sdf_k, np.float32)
    if len(zg) != len(old["zf"]):
        raise SystemExit(f"frame-count mismatch: {len(zg)} from interfaces vs {len(old['zf'])} "
                         "in the cached pairs -- the two files are not from the same pool")
    np.savez(out / "train_pairs.npz", zg=zg, zf=old["zf"], tk=old["tk"], sdf_k=sdf_k,
             fpos=old["fpos"], fval=old["fval"])

    te_old = pickle.load(open(need(EX2_CACHE / "test_enc.pkl"), "rb"))
    te = [[dict(zgeom=enc(f["loops"]).astype(np.float32), zf=fo["zf"], loops=f["loops"],
                time=f["time"], ncomp=f["ncomp"])
           for f, fo in zip(t, to)] for t, to in zip(iface["test"], te_old)]
    pickle.dump(te, open(out / "test_enc.pkl", "wb"))
    print(f"re-encoded {len(zg)} train pairs + {len(te)} test trajectories "
          f"(grid {grid}, order {order}, {BASIS} basis) -> {out}", flush=True)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class Std:
    def __init__(s, a): s.m, s.sd = a.mean(0, keepdims=True), a.std(0, keepdims=True) + 1e-8
    def __call__(s, a): return ((a - s.m) / s.sd).astype(np.float32)


class EvolutionOperator(torch.nn.Module):
    """Time-conditioned ONet: branch product over (z_geom(0), z_f, time embedding[k]).

    geom_head='code'   : MLP(bp) -> order^2 coefficients of z_geom(t_k)
    geom_head='linear' : a single linear read-out bp -> order^2 (no hidden layer)
    The field head p(., t_k) is always a spatial trunk (auxiliary).
    """

    def __init__(s, dg, df, ntime, order, geom_head="code", P=128, H=256, D=3):
        super().__init__()
        s.order = order; s.geom_head = geom_head
        s.bg = FNN([dg] + [H] * D + [P], "relu")
        s.bf = FNN([df] + [H] * D + [P], "relu")
        s.temb = torch.nn.Embedding(ntime, P)
        s.t_field = FNN([2] + [H] * D + [P], "relu")
        s.b_field = torch.nn.Parameter(torch.zeros(1))
        s.geom_mlp = FNN([P] + [order * order], "relu") if geom_head == "linear" \
            else FNN([P] + [H] * D + [order * order], "relu")

    def branch(s, zg, zf, k):
        return s.bg(zg) * s.bf(zf) * s.temb(k)                  # [batch, P]

    def head(s, bp, trunk, pts, bias):
        return (bp.unsqueeze(1) * trunk(pts)).sum(-1) + bias     # [batch, n]

    def field(s, bp, pts):
        return s.head(bp, s.t_field, pts, s.b_field)

    def geom_code(s, bp):
        return s.geom_mlp(bp)                                    # [batch, order^2]


def norm_code(zt):
    """Per-dimension mean, GLOBAL std (so low-frequency magnitude is preserved)."""
    m = zt.mean(0, keepdims=True); sd = float(zt.std()) + 1e-8
    return ((zt - m) / sd).astype(np.float32), (m.astype(np.float32), np.full_like(m, sd, np.float32))


def ckpt_name(head, order, seed):
    return f"evolution_{head}-o{order}-s{seed}.pt"


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(iters, geom_head, order, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    d = np.load(need(EX2_CACHE / "train_pairs.npz"))
    ZG, ZF, TK, SN = d["zg"], d["zf"], d["tk"], d["sdf_k"]
    fpos_a, fval_a = d["fpos"], d["fval"]
    ntime = int(TK.max()) + 1
    print(f"{len(ZG)} train samples (grid {GEOM_GRID}, order {GEOM_ORDER} -> use {order}, "
          f"{ntime} times); head={geom_head} seed={seed}", flush=True)

    sf = Std(ZF)
    mg = ZG.mean(0, keepdims=True).astype(np.float32)            # per-dimension input normalization
    sdg = (ZG.std(0, keepdims=True) + 1e-8).astype(np.float32)
    fmean, fstd = float(fval_a.mean()), float(fval_a.std()) + 1e-12

    tt = lambda a: torch.tensor(a, device=DEV)
    zg = tt(((ZG - mg) / sdg).astype(np.float32)); zf = tt(sf(ZF)); tk = tt(TK)
    fpos = tt(fpos_a); fval = tt(((fval_a - fmean) / fstd).astype(np.float32))

    # geometry target: code of the reference SDF, truncated to order^2, then standardized
    code24 = sdf_to_code(SN.reshape(-1, GEOM_GRID, GEOM_GRID))
    codet = code24.reshape(-1, GEOM_ORDER, GEOM_ORDER)[:, :order, :order].reshape(len(SN), order * order)
    zct, szg = norm_code(codet); zcode = tt(zct)
    fw = tt(freq_weights(order, FREQ_P))

    model = EvolutionOperator(ZG.shape[1], ZF.shape[1], ntime, order, geom_head).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    mse = torch.nn.MSELoss()
    path = out_dir(EX2_CKPT) / ckpt_name(geom_head, order, seed)

    def save():
        torch.save(dict(model=model.state_dict(), ntime=ntime, order=order, geom_head=geom_head,
                        freq_p=FREQ_P, basis=BASIS, sfld=(fmean, fstd), sg=(mg, sdg),
                        sf=(sf.m, sf.sd), szg=szg), path)

    n = len(ZG); t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.randint(0, n, (BATCH,), device=DEV)
        bp = model.branch(zg[idx], zf[idx], tk[idx])
        Lg = (fw * (model.geom_code(bp) - zcode[idx]) ** 2).mean()
        Lf = mse(model.field(bp, fpos[idx]), fval[idx])
        loss = Lg + W_FIELD * Lf
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % max(1, iters // 20) == 0:
            save()
            print(f"  iter {it}: Lg {Lg.item():.4e} Lf {Lf.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    save()
    print(f">>> trained {nparam/1e3:.0f}k params, {iters} iters, {time.time()-t0:.0f}s -> {path.name}",
          flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "encode":
        cache_encode(); return
    if len(sys.argv) > 1 and sys.argv[1] == "reencode":
        reencode(int(sys.argv[2]) if len(sys.argv) > 2 else GEOM_GRID,
                 int(sys.argv[3]) if len(sys.argv) > 3 else GEOM_ORDER); return
    train(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 40000,
          geom_head=sys.argv[2] if len(sys.argv) > 2 else "code",
          order=int(sys.argv[3]) if len(sys.argv) > 3 else 24,
          seed=int(sys.argv[4]) if len(sys.argv) > 4 else 0)


if __name__ == "__main__":
    main()
