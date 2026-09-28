"""Example 3 diagnostic: how the Lipschitz adjacency of a multi-drop frame affects z_gamma.

`encode_surface_function` receives the points of ALL loops stacked into one array and takes one
global cyclic difference to estimate the Lipschitz constant `L`, so the high-percentile slope
includes a few spurious edges that jump from the end of one loop to the start of the next.  This
script measures what that costs, over every multi-loop frame of the encoded pool:

  * the scalar `L`, concatenated vs. each loop closed separately,
  * the resulting code `z_gamma` (identical points and values -- only `L` differs),
  * the decoded surface function at the interface points, and its error against the true surfactant.

Single-loop frames are bit-identical under both adjacencies and are skipped.  A per-frame CSV is
written to `results/adjacency_frames.csv`.  This reads the interface + surfactant data directly,
so it does not need the encoded pool.

Scope: this compares encoder INPUTS (state -> z_gamma).  It is not a comparison of the trained
operator's predictions and does not by itself bound any change in the reported Example-3 errors.

  python diag_adjacency.py
"""

from __future__ import annotations

import csv
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import encoding as ENC                                                  # noqa: E402
from encoding import GRID_ENC, ORDER                                    # noqa: E402
from iae.encoders import decode_surface_function, encode_surface_function  # noqa: E402
from iae.encoders.interface import _curve_lipschitz                     # noqa: E402
from iae.paths import EX3_DATA, need, out_dir                           # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def lipschitz_per_loop(loops, fvals, pct=99.0, safety=1.2):
    """High-percentile neighbour slope with EACH loop closed separately (no inter-loop edges)."""
    slopes = []
    for lp, fv in zip(loops, fvals):
        p = np.asarray(lp, float); v = np.asarray(fv, float)
        dp = np.linalg.norm(np.diff(p, axis=0, append=p[:1]), axis=1) + 1e-9
        dv = np.abs(np.diff(v, append=v[:1]))
        slopes.append(dv / dp)
    return float(np.percentile(np.concatenate(slopes), pct)) * safety + 1e-6


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    rows = []
    for split in ("train", "test"):
        # Read the interface + surfactant data directly and apply the same partial-frame
        # trimming the encoder does, so this needs no encoded pool of its own.
        d = ENC.clean_partials(pickle.load(open(need(EX3_DATA / f"traj_{split}.pkl"), "rb")))
        box = d["box"]
        for tr in d["trajs"]:
            for fr in tr:
                loops, fvals = fr["loops"], fr["fvals"]
                if len(loops) < 2:
                    continue                                # single loop: concat == per-loop
                pts = np.vstack([np.asarray(l, float) for l in loops])
                vals = np.concatenate([np.asarray(v, float) for v in fvals])
                L_c = _curve_lipschitz(pts, vals)
                L_p = lipschitz_per_loop(loops, fvals)
                z_c = encode_surface_function(pts, vals, box, grid_n=GRID_ENC, order=ORDER)
                z_p = encode_surface_function(pts, vals, box, grid_n=GRID_ENC, order=ORDER, L=L_p)
                f_c = decode_surface_function(z_c, pts, box)
                f_p = decode_surface_function(z_p, pts, box)
                rows.append(dict(split=split, trajectory=int(fr.get("traj", -1)),
                                 time=float(fr.get("t", -1)), nloops=len(loops),
                                 L_concat=L_c, L_perloop=L_p, relL_signed=(L_p - L_c) / L_c,
                                 rel_z=rel_l2(z_p, z_c), rel_f=rel_l2(f_p, f_c),
                                 err_concat=rel_l2(f_c, vals), err_perloop=rel_l2(f_p, vals)))
        print(f"{split}: processed", flush=True)
    if not rows:
        raise SystemExit("no multi-loop frames found")

    out = out_dir(HERE / "results") / "adjacency_frames.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    A = {k: np.array([r[k] for r in rows]) for k in
         ("relL_signed", "rel_z", "rel_f", "err_concat", "err_perloop")}
    print(f"\nwrote {out}  ({len(rows)} multi-loop frames; single-loop frames are identical "
          "under both adjacencies and were skipped)\n")

    q = lambda a: (f"median {np.median(np.abs(a)):.3e} | 90th {np.percentile(np.abs(a), 90):.3e} "
                   f"| max {np.abs(a).max():.3e}")
    print("relative change induced by the per-loop adjacency, over multi-loop frames:")
    print(f"  L         |L_p - L_c| / L_c      : {q(A['relL_signed'])}")
    print(f"  z_gamma   ||z_p - z_c|| / ||z_c||: {q(A['rel_z'])}")
    print(f"  decoded f ||f_p - f_c|| / ||f_c||: {q(A['rel_f'])}")
    e_c, e_p = A["err_concat"].mean(), A["err_perloop"].mean()
    print("\nreconstruction rel-L2 against the true surfactant at the interface points:")
    print(f"  concatenated (shipped): mean {e_c:.4f}  median {np.median(A['err_concat']):.4f}")
    print(f"  per-loop              : mean {e_p:.4f}  median {np.median(A['err_perloop']):.4f}")
    print(f"  per-loop is closer to truth on {(A['err_perloop'] < A['err_concat']).mean()*100:.1f}% "
          f"of frames; mean(err_perloop - err_concat) = {e_p - e_c:+.6f}")


if __name__ == "__main__":
    main()
