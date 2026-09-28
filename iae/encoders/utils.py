"""Shared helpers for the encoders (Legendre basis, quadrature constants, polygon->SDF)."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legvander, leggauss
from matplotlib.path import Path


def legendre_normalized(x: np.ndarray, n: int) -> np.ndarray:
    """L2([0,1])-orthonormal Legendre basis up to degree n-1, evaluated at x."""
    x = np.asarray(x)
    return legvander(2.0 * x - 1.0, n - 1) * np.sqrt(2.0 * np.arange(n) + 1.0)


# 3-point barycentric rule (degree-2 exact) for triangle integration.
TRI_BARY = np.array([[1 / 6, 1 / 6, 2 / 3], [1 / 6, 2 / 3, 1 / 6], [2 / 3, 1 / 6, 1 / 6]])
TRI_WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])

# 5-point Gauss-Legendre rule on [0,1] for boundary edges.
_GX, _GW = leggauss(5)
EDGE_T = 0.5 * (_GX + 1.0)
EDGE_W = 0.5 * _GW


# ---- polygon -> signed distance field (pure numpy) ----

def poly_sdf(loop: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Signed distance (negative inside) from points ``P[N,2]`` to a closed polygon ``loop[M,2]``."""
    A = loop
    B = np.roll(loop, -1, axis=0)
    AB = B - A
    AP = P[:, None, :] - A[None, :, :]
    t = np.clip((AP * AB[None]).sum(-1) / ((AB * AB).sum(-1)[None] + 1e-30), 0.0, 1.0)
    proj = A[None] + t[..., None] * AB[None]
    d = np.linalg.norm(P[:, None, :] - proj, axis=-1).min(1)
    inside = Path(loop).contains_points(P)
    return np.where(inside, -d, d)


def union_sdf(loops, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Signed distance of the union of closed ``loops`` sampled on the meshgrid ``(X, Y)``.

    Negative inside any loop; ``phi`` has the shape of ``X`` (== shape of ``Y``)."""
    P = np.column_stack([X.ravel(), Y.ravel()])
    phi = np.full(len(P), 1e9)
    for loop in loops:
        phi = np.minimum(phi, poly_sdf(loop, P))
    return phi.reshape(X.shape)
