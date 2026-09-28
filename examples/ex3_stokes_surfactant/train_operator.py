"""Example 3: the joint interface + surfactant operator for Stokes flow.

Time-conditioned, like Example 2, but with TWO code heads -- the state is a pair
(interface, function on the interface) and both are predicted from the initial
state alone:

    input   z_geom(0), z_gamma(0), time index k
    branch  b_g(z_geom(0)) * b_gam(z_gamma(0)) * E[k]
    heads   geom_mlp(bp) -> z_geom(t_k),   gam_mlp(bp) -> z_gamma(t_k)
    loss    standardized-coefficient MSE on both codes

At evaluation the interfaces are the zero level set of the decoded z_geom and the
surfactant is the decoded z_gamma read on those interfaces, so drop count and
surface values both come out of the decoder.

  python train_operator.py encode           # once: encode trajectories, cache code pairs
  python train_operator.py [iters] [seed]   # train  -> checkpoints/ex3_stokes_surfactant/
"""

from __future__ import annotations

import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import encoding as ENC                                                  # noqa: E402
from iae.paths import EX3_CACHE, EX3_CKPT, EX3_DATA, need, out_dir      # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256


class FNN(torch.nn.Module):
    """Plain GELU MLP (the Example-3 branches and heads)."""

    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.GELU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SurfOperator(torch.nn.Module):
    def __init__(self, dg, dgam, ntime, order, P=128, H=256, D=3):
        super().__init__()
        self.order = order
        self.bg = FNN([dg] + [H] * D + [P])
        self.bgam = FNN([dgam] + [H] * D + [P])
        self.temb = torch.nn.Embedding(ntime, P)
        self.geom_mlp = FNN([P] + [H] * D + [order * order])
        self.gam_mlp = FNN([P] + [H] * D + [order * order])

    def forward(self, zg0, zgam0, k):
        bp = self.bg(zg0) * self.bgam(zgam0) * self.temb(k)
        return self.geom_mlp(bp), self.gam_mlp(bp)


class Std:
    """Per-dimension mean, global std (keeps the low-frequency coefficient magnitude)."""

    def __init__(self, a):
        self.m = a.mean(0, keepdims=True).astype(np.float32)
        self.sd = np.float32(a.std() + 1e-8)

    def fwd(self, a):
        return ((a - self.m) / self.sd).astype(np.float32)


def do_encode():
    t0 = time.time(); out = out_dir(EX3_CACHE)
    for split in ("train", "test"):
        data = pickle.load(open(need(EX3_DATA / f"traj_{split}.pkl"), "rb"))
        ENC.encode_trajs(data)
        pickle.dump(data, open(out / f"traj_{split}_enc.pkl", "wb"))
        P = ENC.build_pairs(data)
        np.savez(out / f"pairs_{split}.npz", **{k: v for k, v in P.items() if k != "ntime"},
                 ntime=P["ntime"], box=np.array(data["box"]))
        print(f"  {split}: {len(P['tk'])} pairs, ntime={P['ntime']} ({time.time()-t0:.0f}s)", flush=True)
    print(f"encoded (order {ENC.ORDER}, grid {ENC.GRID_ENC}) -> {out}")


def train(iters, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = np.load(need(EX3_CACHE / "pairs_train.npz"))
    order = ENC.ORDER; ntime = int(tr["ntime"])
    sZg, sZgam = Std(tr["zg0"]), Std(tr["zgam0"])           # input scalers
    sTg, sTgam = Std(tr["zgt"]), Std(tr["zgamt"])           # target scalers
    Zg = torch.tensor(sZg.fwd(tr["zg0"]), device=DEV)
    Zgam = torch.tensor(sZgam.fwd(tr["zgam0"]), device=DEV)
    K = torch.tensor(tr["tk"], device=DEV)
    Yg = torch.tensor(sTg.fwd(tr["zgt"]), device=DEV)
    Ygam = torch.tensor(sTgam.fwd(tr["zgamt"]), device=DEV)
    n = len(K)

    model = SurfOperator(Zg.shape[1], Zgam.shape[1], ntime, order).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.randint(0, n, (min(BATCH, n),), device=DEV)
        pg, pgam = model(Zg[idx], Zgam[idx], K[idx])
        loss = ((pg - Yg[idx]) ** 2).mean() + ((pgam - Ygam[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % max(1, iters // 10) == 0:
            print(f"  iter {it}: loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    path = out_dir(EX3_CKPT) / f"surf_operator-s{seed}.pt"
    torch.save(dict(model=model.state_dict(), order=order, ntime=ntime,
                    sZg=(sZg.m, sZg.sd), sZgam=(sZgam.m, sZgam.sd),
                    sTg=(sTg.m, sTg.sd), sTgam=(sTgam.m, sTgam.sd),
                    box=tr["box"].tolist()), path)
    print(f">>> saved {path.name} (order {order}, ntime {ntime}, {n} pairs)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "encode":
        do_encode()
    else:
        train(int(sys.argv[1]) if len(sys.argv) > 1 else 30000,      # paper: 3e4 steps
              int(sys.argv[2]) if len(sys.argv) > 2 else 0)
