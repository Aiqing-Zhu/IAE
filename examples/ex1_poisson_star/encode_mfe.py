"""Example 1, step 1a: query points, targets, and the MFE baseline codes.

Builds, once, the two files every Example-1 training run reads:

    mfe_train.npz   pos, yv                  300 random (node, u(node)) pairs per sample
                    mfe_zgeom, mfe_zgeom1d   2D- and 1D-MFE geometry codes
                    mfe_zk, mfe_zf, mfe_zg   Legendre moments of k, f, g
    mfe_test.npz    points, u, notch + the same codes, full meshes kept

The 1D-MFE geometry code is the boundary moment of the constant function 1, i.e.
the same edge quadrature with g set to 1; the 2D-MFE code is the domain moment of
the indicator of Omega.  Both are computed on the GPU in one pass.

Samples are read in the fixed order raw_train.npz -> raw_train_b.npz -> raw_test.npz
with no shuffling; encode_iae.py uses the same order, so sample i denotes the same
domain in every encoding.

  python encode_mfe.py
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iae.encoders import gpu                                    # noqa: E402
from iae.paths import EX1_CACHE, EX1_DATA, need, out_dir        # noqa: E402

DEV = "cuda"
ORDER, NUM_LOC = 12, 300


def load(path):
    """Unpack one ragged .npz of meshed samples into a list of dicts."""
    d = np.load(path); num = int(d["num"])
    P, U, NO, pp = d["points"], d["u"], d["notch"], d["point_ptr"]
    T, BE, K, F, G = d["triangles"], d["bedges"], d["k"], d["f"], d["g"]; tp, bp = d["tri_ptr"], d["bedge_ptr"]
    return [dict(points=P[pp[i]:pp[i + 1]], u=U[pp[i]:pp[i + 1]], notch=bool(NO[i]),
                 triangles=T[tp[i]:tp[i + 1]] - pp[i], bedges=BE[bp[i]:bp[i + 1]] - pp[i],
                 k=K[pp[i]:pp[i + 1]], f=F[pp[i]:pp[i + 1]], g=G[pp[i]:pp[i + 1]]) for i in range(num)]


def mfe_codes(s):
    """(zgeom_2d, zk, zf, zg, zgeom_1d) for a list of samples."""
    f = gpu.encode_batch(s, encoder="mfe", device=DEV, mfe_mode=ORDER)   # real g -> zg = boundary moment of g
    for x in s:
        x["g"] = np.ones(len(x["points"]))                               # set g == 1
    f1 = gpu.encode_batch(s, encoder="mfe", device=DEV, mfe_mode=ORDER)  # now zg = boundary moment of 1 = 1D geometry
    return f["zgeom"], f["zk"], f["zf"], f["zg"], f1["zg"]


def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    out = out_dir(EX1_CACHE)
    pos, yv, zmfe = [], [], {k: [] for k in ("zgeom", "zk", "zf", "zg", "zgeom1d")}
    for fn in ("raw_train.npz", "raw_train_b.npz"):
        s = load(need(EX1_DATA / fn))
        for x in s:
            idx = rng.choice(len(x["points"]), NUM_LOC, replace=len(x["points"]) < NUM_LOC)
            pos.append(x["points"][idx]); yv.append(x["u"][idx])
        for k, v in zip(zmfe, mfe_codes(s)):
            zmfe[k].append(v)
        print(f"  {fn}: {len(s)} samples ({time.time()-t0:.0f}s)", flush=True); del s
    np.savez(out / "mfe_train.npz", pos=np.array(pos, np.float32), yv=np.array(yv, np.float32),
             **{f"mfe_{k}": np.concatenate(v) for k, v in zmfe.items()})

    st = load(need(EX1_DATA / "raw_test.npz"))                 # test: keep the full meshes
    zg2, zk, zf, zg, zg1 = mfe_codes(st)
    np.savez(out / "mfe_test.npz",
             points=np.array([x["points"].astype(np.float32) for x in st], dtype=object),
             u=np.array([x["u"].astype(np.float32) for x in st], dtype=object),
             notch=np.array([x["notch"] for x in st]),
             mfe_zgeom=zg2, mfe_zk=zk, mfe_zf=zf, mfe_zg=zg, mfe_zgeom1d=zg1)
    print(f"wrote {out}/mfe_train.npz, mfe_test.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
