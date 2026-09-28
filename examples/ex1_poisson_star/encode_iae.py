"""Example 1, step 1b: the IAE codes.

Encodes every sample with `iae.encoders.encode_gext`:

    z_geom   signed-distance field of dOmega, projected on the tensor basis
    z_g      boundary datum g: McShane-Whitney extension to the box, then projected
    z_k,z_f  interior fields: Legendre moments over Omega

Writes `iae_{train,test}_{basis}.npz` with the four codes plus the per-sample
notch flag.  Sample order matches encode_mfe.py exactly.

  python encode_iae.py [n_proc=16] [basis=cosine|legendre]
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"                                        # one BLAS thread per worker

import multiprocessing as mp                                    # noqa: E402
import pathlib                                                  # noqa: E402
import sys                                                      # noqa: E402
import time                                                     # noqa: E402

import numpy as np                                              # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iae.encoders import encode_gext                            # noqa: E402
from iae.paths import EX1_CACHE, EX1_DATA, need, out_dir        # noqa: E402

NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 16
BASIS = sys.argv[2] if len(sys.argv) > 2 else "cosine"
ORDER, GRID = 12, 24
_D = ORDER * ORDER


def _load(path):
    d = np.load(path); num = int(d["num"])
    P, T, BE, K, F, G, NO = d["points"], d["triangles"], d["bedges"], d["k"], d["f"], d["g"], d["notch"]
    pp, tp, bp = d["point_ptr"], d["tri_ptr"], d["bedge_ptr"]
    return [dict(points=P[pp[i]:pp[i + 1]], triangles=T[tp[i]:tp[i + 1]] - pp[i],
                 bedges=BE[bp[i]:bp[i + 1]] - pp[i], k=K[pp[i]:pp[i + 1]], f=F[pp[i]:pp[i + 1]],
                 g=G[pp[i]:pp[i + 1]], notch=bool(NO[i])) for i in range(num)]


def _one(s):
    zgeom, zk, zf, zg = encode_gext(
        s["points"].astype(float), s["triangles"], s["bedges"], s["k"], s["f"], s["g"],
        order=ORDER, grid=GRID, geom_order=ORDER, geom_grid=GRID, basis=BASIS, mfe_mode=ORDER)
    return np.concatenate([zgeom, zk, zf, zg]).astype(np.float32)


def _encode(samples):
    with mp.Pool(NPROC) as pool:
        return np.array(pool.map(_one, samples), np.float32)


def _save(path, codes, notch):
    np.savez(path, zgeom=codes[:, :_D], zk=codes[:, _D:2 * _D], zf=codes[:, 2 * _D:3 * _D],
             zg=codes[:, 3 * _D:], notch=notch)


def main():
    t0 = time.time()
    out = out_dir(EX1_CACHE)
    parts, notch = [], []
    for fn in ("raw_train.npz", "raw_train_b.npz"):             # one file at a time to bound memory
        s = _load(need(EX1_DATA / fn))
        parts.append(_encode(s)); notch.append(np.array([x["notch"] for x in s]))
        print(f"  encoded {fn}: {len(s)} samples ({time.time()-t0:.0f}s)", flush=True); del s
    _save(out / f"iae_train_{BASIS}.npz", np.concatenate(parts), np.concatenate(notch))

    s = _load(need(EX1_DATA / "raw_test.npz"))
    _save(out / f"iae_test_{BASIS}.npz", _encode(s), np.array([x["notch"] for x in s]))
    print(f"wrote {out}/iae_{{train,test}}_{BASIS}.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
