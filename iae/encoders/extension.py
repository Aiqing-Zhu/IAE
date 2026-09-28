"""Lipschitz (McShane-Whitney) extension + orthonormal basis projection.

This module holds the two primitives the Interface Autoencoder is built from.

1. **Extension.**  A function known only on a lower-dimensional set (a boundary
   ``dOmega``, an interface ``Gamma``) is extended to the whole bounding box by an
   inf-convolution with the cone ``L|.|``.  For an ``L``-Lipschitz function the
   McShane-Whitney extensions

       inf (maximal):  G_hi(xi) = min_i ( v_i + L |xi - y_i| )
       sup (minimal):  G_lo(xi) = max_i ( v_i - L |xi - y_i| )
       mid          :  (G_hi + G_lo) / 2                        (default)

   all interpolate the data and are ``L``-Lipschitz; every ``L``-Lipschitz
   extension lies between the first two.  For a *function on the interface* the
   midpoint is used: it cancels the one-sided kink bias of the two envelopes and
   is therefore friendlier to a smooth basis.  For the *geometry* the one-sided
   ``inf`` extension of ``g = 0`` with ``L = 1`` is used -- it is exactly the
   distance to the interface (the midpoint of the two envelopes of ``g = 0``
   vanishes identically) -- and is then signed by the inside / outside test, which
   is what :func:`signed_distance_from_edges` computes.

2. **Projection.**  The extended field is sampled on a regular grid over the box
   and projected onto a tensor-product basis that is orthonormal on ``[0, 1]``.

   The midpoint rule used here is *exact* for the cosine basis -- on this grid it
   is a DCT-II, whose discrete Gram matrix is the identity to machine precision --
   so decoding inverts encoding.  It is only second-order accurate for Legendre
   products, whose discrete Gram matrix is far from the identity at ``grid_n = 2
   order`` (about 0.2 off-diagonal).  Legendre coefficients obtained here are
   therefore usable as *features* (Example 1 compares them against the cosine ones
   as encoders) but must not be fed back through
   :func:`basis_reconstruct` unless the grid is much finer than the usual
   ``grid_n >= 2 order`` rule.  Everything that decodes in this repository uses the
   cosine basis.

So shape and surface function live in the same coefficient space as the MIONet
trunk's query point.
"""

from __future__ import annotations

import numpy as np

from .utils import legendre_normalized


def mcshane_extend(query: np.ndarray, src_pts: np.ndarray, src_vals: np.ndarray,
                   L: float, mode: str = "mid", chunk: int = 512) -> np.ndarray:
    """Lipschitz extension evaluated at `query` points. mode in {inf, sup, mid}."""
    out = []
    for i in range(0, len(query), chunk):
        d = np.linalg.norm(query[i : i + chunk, None, :] - src_pts[None, :, :], axis=2)  # [c, N]
        if mode == "inf":
            out.append((src_vals[None, :] + L * d).min(axis=1))
        elif mode == "sup":
            out.append((src_vals[None, :] - L * d).max(axis=1))
        elif mode == "mid":
            hi = (src_vals[None, :] + L * d).min(axis=1)
            lo = (src_vals[None, :] - L * d).max(axis=1)
            out.append(0.5 * (hi + lo))
        else:
            raise ValueError(f"unknown extension mode {mode}")
    return np.concatenate(out)


def basis_1d(u: np.ndarray, order: int, kind: str = "cosine") -> np.ndarray:
    """Orthonormal-on-[0,1] 1-D basis, evaluated at u; returns [..., order]."""
    if kind == "cosine":
        m = np.arange(order)
        b = np.cos(np.pi * u[..., None] * m)
        return b * np.where(m == 0, 1.0, np.sqrt(2.0))  # L2([0,1])-orthonormal
    if kind == "legendre":
        return legendre_normalized(u, order)
    raise ValueError(f"unknown basis {kind}")


def extend_and_project(src_pts: np.ndarray, src_vals: np.ndarray, L: float,
                       box=(0.0, 1.0, 0.0, 1.0), grid_n: int = 24, order: int = 12,
                       basis: str = "cosine", mode: str = "mid") -> np.ndarray:
    """Encode a function: Lipschitz-extend to a grid, then tensor-basis project.

    Returns a length-order^2 coefficient vector (the finite-dim representation).
    """
    x_lo, x_hi, y_lo, y_hi = box
    a = (np.arange(grid_n) + 0.5) / grid_n  # normalized cell centers in [0,1]
    xx, yy = np.meshgrid(x_lo + a * (x_hi - x_lo), y_lo + a * (y_hi - y_lo), indexing="ij")
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    field = mcshane_extend(grid, src_pts, src_vals, L, mode).reshape(grid_n, grid_n)

    return _project_grid(field, a, order, basis).reshape(-1)


def _project_grid(field_grid: np.ndarray, a: np.ndarray, order: int, basis: str) -> np.ndarray:
    """Tensor-basis projection of a field sampled on a regular grid (midpoint rule).

    Exact for the cosine basis (a DCT-II on this grid); only O(h^2) for Legendre -- see the
    module docstring before using Legendre coefficients on the decoding side."""
    b = basis_1d(a, order, basis)  # [grid_n, order]
    h = 1.0 / len(a)
    return np.einsum("ab,ai,bj->ij", field_grid, b, b, optimize=True) * (h * h)


def basis_reconstruct(coeffs: np.ndarray, points: np.ndarray,
                      box=(0.0, 1.0, 0.0, 1.0), basis: str = "cosine") -> np.ndarray:
    """Evaluate the tensor-basis expansion (inverse of extend_and_project) at points."""
    order = int(round(len(coeffs) ** 0.5))
    x_lo, x_hi, y_lo, y_hi = box
    ux = (points[:, 0] - x_lo) / (x_hi - x_lo)
    uy = (points[:, 1] - y_lo) / (y_hi - y_lo)
    bx, by = basis_1d(ux, order, basis), basis_1d(uy, order, basis)
    return np.einsum("ni,nj,ij->n", bx, by, coeffs.reshape(order, order), optimize=True)


# --------------------------------------------------------------------------- #
# Signed distance to a boundary given as an unordered EDGE set.  phi < 0 inside,
# = 0 on the boundary, > 0 outside.
# --------------------------------------------------------------------------- #
def _segment_distance(query: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Point-to-segment distance for ALL M points x E edges via BLAS matrix products
    # (Q @ A.T, Q @ AB.T) -- no [M,E,2] temporary.  d2 = |AQ|^2 - 2t(AQ.AB) + t^2|AB|^2.
    # float64: the |Q-A|^2 = |Q|^2 - 2 Q.A + |A|^2 form cancels in float32 for small
    # distances (coords ~0.5), so keep double precision (GEMM is fast either way).
    query = np.asarray(query, np.float64)
    A = np.asarray(A, np.float64)
    AB = B - A  # [E,2]
    AB2 = np.einsum("ed,ed->e", AB, AB) + 1e-20            # [E]   |AB|^2
    A_AB = np.einsum("ed,ed->e", A, AB)                    # [E]   A . AB
    sqA = np.einsum("ed,ed->e", A, A)                      # [E]   |A|^2
    sqQ = np.einsum("md,md->m", query, query)              # [M]   |Q|^2
    AQ_AB = query @ AB.T - A_AB[None, :]                   # [M,E]  (Q-A).AB   (GEMM)
    AQ2 = sqQ[:, None] - 2.0 * (query @ A.T) + sqA[None, :]  # [M,E]  |Q-A|^2   (GEMM)
    t = np.clip(AQ_AB / AB2[None, :], 0.0, 1.0)            # [M,E]
    d2 = AQ2 - 2.0 * t * AQ_AB + (t * t) * AB2[None, :]    # [M,E]
    return np.sqrt(np.maximum(d2, 0.0)).min(axis=1)


def _inside_edges(query: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Even-odd ray casting over a boundary EDGE set (no ordering required)."""
    x, y = query[:, 0], query[:, 1]
    x1, y1 = A[:, 0][None, :], A[:, 1][None, :]
    x2, y2 = B[:, 0][None, :], B[:, 1][None, :]
    straddle = (y1 > y[:, None]) != (y2 > y[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = (x2 - x1) * (y[:, None] - y1) / (y2 - y1) + x1
    return (np.sum(straddle & (x[:, None] < xc), axis=1) % 2) == 1


def signed_distance_from_edges(query: np.ndarray, points: np.ndarray, edges: np.ndarray,
                               chunk: int = 8192) -> np.ndarray:
    """Signed distance to a boundary given as EDGES (no loop ordering needed)."""
    A, B = points[edges[:, 0]], points[edges[:, 1]]
    out = []
    for i in range(0, len(query), chunk):
        q = query[i : i + chunk]
        d = _segment_distance(q, A, B)
        out.append(np.where(_inside_edges(q, A, B), -d, d))
    return np.concatenate(out)
