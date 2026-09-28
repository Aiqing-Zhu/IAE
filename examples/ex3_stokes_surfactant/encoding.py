"""Example 3: encode a (interface, surfactant) state and build the training pairs.

Each recorded frame holds one or more closed interfaces together with the
surfactant concentration sampled at their points.  The IAE turns that into two
codes of the same length:

    z_geom   = code of the union signed-distance field of the interfaces
    z_gamma  = code of the McShane-Whitney extension of the surfactant

The operator predicts BOTH from the initial-frame codes plus a discrete time index,
so the decoder recovers the interfaces (and hence the drop count) and the
surfactant on them from the predicted codes alone.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iae.encoders import encode_interface, encode_surface_function      # noqa: E402

ORDER = 32          # code side length: z_geom, z_gamma are each ORDER^2
GRID_ENC = 96       # projection grid (>= 2*ORDER, so the projection does not alias)


def clean_partials(data):
    """Drop the LAST frame of any trajectory that ended early (blow-up): the pre-blow-up
    frame can be slightly corrupted by the elevated velocity.  Full-length trajectories
    are untouched."""
    T = data.get("T"); dts = data.get("dt_sample", 1.0)
    if T is None:
        return data
    for i, tr in enumerate(data["trajs"]):
        if tr and tr[-1]["t"] < T - dts / 2 and len(tr) > 3:
            data["trajs"][i] = tr[:-1]
    return data


def encode_frame(frame, box):
    """(loops, fvals) frame -> (z_geom, z_gamma), each ORDER^2."""
    z_geom = encode_interface(frame["loops"], box, grid_n=GRID_ENC, order=ORDER)
    pts = np.vstack([np.asarray(l, float) for l in frame["loops"]])
    vals = np.concatenate([np.asarray(v, float) for v in frame["fvals"]])
    z_gam = encode_surface_function(pts, vals, box, grid_n=GRID_ENC, order=ORDER)
    return z_geom.astype(np.float32), z_gam.astype(np.float32)


def encode_trajs(data):
    """Attach zgeom/zgam to every frame of every trajectory (in place); trims partials first."""
    clean_partials(data)
    box = data["box"]
    for tr in data["trajs"]:
        for fr in tr:
            fr["zgeom"], fr["zgam"] = encode_frame(fr, box)
    return data


def build_pairs(data):
    """Time-conditioned pairs: (frame-0 codes, time index k) -> frame-k codes.  k = 0..K-1
    maps to the (k+1)-th recorded time; the input is ALWAYS the initial frame."""
    zg0, zgam0, tk, zgt, zgamt = [], [], [], [], []
    for tr in data["trajs"]:
        z0g, z0gam = tr[0]["zgeom"], tr[0]["zgam"]
        for k in range(1, len(tr)):
            zg0.append(z0g); zgam0.append(z0gam); tk.append(k - 1)
            zgt.append(tr[k]["zgeom"]); zgamt.append(tr[k]["zgam"])
    return dict(zg0=np.array(zg0), zgam0=np.array(zgam0), tk=np.array(tk, np.int64),
                zgt=np.array(zgt), zgamt=np.array(zgamt),
                ntime=max((len(t) - 1 for t in data["trajs"]), default=0))
