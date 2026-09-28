"""Interface Autoencoder (IAE): spectral encode / decode of an interface and of a
function defined on it.

Everything here is one recipe -- *extend to the bounding box, then project onto a
truncated orthonormal tensor basis* (see :mod:`iae.encoders.extension`) -- applied
to three kinds of object:

===========================  =================================================
object                       encoding
===========================  =================================================
region / interface `Gamma`   signed-distance field of `Gamma` (the one-sided
                             `(g=0, L=1)` Lipschitz extension, signed) -> `z_geom`
function on `Gamma`          McShane-Whitney mid-extension -> `z_gamma`
domain field `k`, `f`        Legendre moments over `Omega` (the MFE rule; cheap
                             and exact on the mesh)
===========================  =================================================

Decoding is resolution-free: evaluate the truncated series anywhere.  The
interface is recovered as the **zero level set** of the decoded signed-distance
field (marching squares), so the number of connected components -- the topology --
is an output of the decoder, not an input.  That is what lets the learned
operators pass through merges.

Public entry points
-------------------
``encode_state`` / ``decode_state``
    the full autoencoder on a state `(Gamma, f)`; used by Examples 2 and 3.
``encode_interface`` / ``decode_interface``
    geometry only.
``encode_surface_function`` / ``decode_surface_function``
    a function sampled on interface points.
``encode_gext``
    the Example-1 variable-domain Poisson encoder: interface SDF + boundary datum
    `g` by McShane extension, interior fields `k, f` by moments.
"""

from __future__ import annotations

import numpy as np

from .extension import (_project_grid, basis_1d, basis_reconstruct, extend_and_project,
                        signed_distance_from_edges)
from .utils import EDGE_T, EDGE_W, TRI_BARY, TRI_WEIGHTS


# --------------------------------------------------------------------------- #
# Moment rules for functions given on a mesh (basis is a hyperparameter).
# --------------------------------------------------------------------------- #
def domain_moment(points, triangles, field, basis: str, order: int) -> np.ndarray:
    """Domain (triangle) moments z_ij = int_Omega field * phi_i(x) phi_j(y) dOmega."""
    verts = points[triangles]  # [M, 3, 2]
    vec1, vec2 = verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0]
    areas = 0.5 * np.abs(vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0])
    quad = np.einsum("qb,mbd->mqd", TRI_BARY, verts)  # [M, 3, 2]
    bx = basis_1d(quad[..., 0], order, basis)  # [M, 3, order]
    by = basis_1d(quad[..., 1], order, basis)
    warea = areas[:, None] * TRI_WEIGHTS[None, :]
    field_q = field[triangles] @ TRI_BARY.T  # [M, 3]
    return np.einsum("mqi,mqj,mq,mq->ij", bx, by, field_q, warea, optimize=True).reshape(-1)


def boundary_moment(points, bedges, g, basis: str, order: int) -> np.ndarray:
    """Boundary (edge) moments of g: int_dOmega g * phi_i(x) phi_j(y) ds (5-pt Gauss)."""
    edge_pts = points[bedges]  # [B, 2, 2]
    g_edge = g[bedges]  # [B, 2]
    length = np.linalg.norm(edge_pts[:, 1] - edge_pts[:, 0], axis=1)
    t = EDGE_T  # [Q]
    quad = (1 - t[None, :, None]) * edge_pts[:, 0][:, None] + t[None, :, None] * edge_pts[:, 1][:, None]  # [B,Q,2]
    g_quad = (1 - t[None, :]) * g_edge[:, 0][:, None] + t[None, :] * g_edge[:, 1][:, None]  # [B, Q]
    wedge = EDGE_W[None, :] * length[:, None]  # [B, Q]
    bx = basis_1d(quad[..., 0], order, basis)
    by = basis_1d(quad[..., 1], order, basis)
    return np.einsum("bqi,bqj,bq,bq->ij", bx, by, g_quad, wedge, optimize=True).reshape(-1)


def sdf_geometry_code(points, triangles, bedges, basis: str = "cosine", order: int = 12,
                      mode: str = "dct", grid_n: int = 24,
                      box=(0.0, 1.0, 0.0, 1.0)) -> np.ndarray:
    """Encode a region by the basis coefficients of its signed-distance field.

    mode: 'dct' (SDF on a regular grid_n x grid_n grid, then basis projection --
    decoupled from the mesh, and the cheapest) or 'moment' (domain quadrature of
    the nodal SDF over the mesh)."""
    if mode == "dct":
        x_lo, x_hi, y_lo, y_hi = box
        a = (np.arange(grid_n) + 0.5) / grid_n
        xx, yy = np.meshgrid(x_lo + a * (x_hi - x_lo), y_lo + a * (y_hi - y_lo), indexing="ij")
        sdf = signed_distance_from_edges(np.column_stack([xx.ravel(), yy.ravel()]), points, bedges)
        return _project_grid(sdf.reshape(grid_n, grid_n), a, order, basis).reshape(-1)
    if mode == "moment":
        sdf = signed_distance_from_edges(points, points, bedges)  # nodal level-set field
        return domain_moment(points, triangles, sdf, basis, order)
    raise ValueError(f"unknown geometry coefficient mode {mode}")


def _lip_edges(points, bedges, g, pct: float = 99.0, safety: float = 1.2) -> float:
    """Lipschitz constant of a boundary datum, from boundary-edge slopes |dg|/|dp|."""
    A, B = points[bedges[:, 0]], points[bedges[:, 1]]
    dp = np.linalg.norm(A - B, axis=1) + 1e-9
    return float(np.percentile(np.abs(g[bedges[:, 0]] - g[bedges[:, 1]]) / dp, pct)) * safety + 1e-6


def encode_gext(points, triangles, bedges, k, f, g, *, basis: str = "cosine", order: int = 12,
                grid: int = 24, box=(0.0, 1.0, 0.0, 1.0), geom_order: int = 12, geom_grid: int = 24,
                mfe_mode: int = 12):
    """Example-1 encoder for a variable-domain elliptic problem.

    Geometry is the SDF code; the BOUNDARY datum `g` uses the McShane extension of
    the interface-function recipe; the DOMAIN fields `k, f` use Legendre moments
    over `Omega` (cheaper than extending a field that already fills the domain).
    Returns `(z_geom, z_k, z_f, z_g)`."""
    pts = np.asarray(points, float)
    zk = domain_moment(pts, triangles, np.asarray(k, float), "legendre", mfe_mode)
    zf = domain_moment(pts, triangles, np.asarray(f, float), "legendre", mfe_mode)
    gg = np.asarray(g, float)
    bn = np.unique(np.asarray(bedges).ravel())
    zg = extend_and_project(pts[bn], gg[bn], _lip_edges(pts, bedges, gg), box, grid, order, basis)
    z_geom = sdf_geometry_code(pts, triangles, bedges, basis, geom_order, "dct", geom_grid, box)
    return z_geom, zk, zf, zg


# =========================================================================== #
# Interface + surface function: the autoencoder used by Examples 2 and 3.
# =========================================================================== #
def _loops_to_edges(loops):
    """Stack closed loops (each (Ni,2) ordered points) into one (points, edges) set,
    each loop closed by an edge from its last point back to its first."""
    pts_all, edges, off = [], [], 0
    for lp in loops:
        lp = np.asarray(lp, float)
        if len(lp) < 2:
            continue
        n = len(lp); idx = np.arange(n) + off
        pts_all.append(lp)
        edges.append(np.column_stack([idx, np.roll(idx, -1)]))
        off += n
    return np.vstack(pts_all), np.vstack(edges)


def encode_interface(loops, box, *, grid_n: int = 64, order: int = 12, basis: str = "cosine"):
    """DCT code of the interface: project the union signed-distance field (negative inside
    the loops) onto the tensor cosine basis over `box`.  `loops`: list of (Ni,2) closed
    curves.  Returns z_geom (order^2,)."""
    pts, edges = _loops_to_edges(loops)
    x_lo, x_hi, y_lo, y_hi = box
    a = (np.arange(grid_n) + 0.5) / grid_n
    xx, yy = np.meshgrid(x_lo + a * (x_hi - x_lo), y_lo + a * (y_hi - y_lo), indexing="ij")
    sdf = signed_distance_from_edges(np.column_stack([xx.ravel(), yy.ravel()]), pts, edges)
    return _project_grid(sdf.reshape(grid_n, grid_n), a, order, basis).reshape(-1)


def decode_interface(z_geom, box, *, grid_n: int = 240, basis: str = "cosine", level: float = 0.0):
    """Reconstruct interface loops as the zero level set of the decoded SDF (marching squares).
    Returns a list of (Ni,2) loops (topology = number of loops recovered)."""
    from contourpy import contour_generator
    x_lo, x_hi, y_lo, y_hi = box
    X = np.linspace(x_lo, x_hi, grid_n); Y = np.linspace(y_lo, y_hi, grid_n)
    XX, YY = np.meshgrid(X, Y)                                       # (grid_n, grid_n), 'xy'
    sdf = basis_reconstruct(z_geom, np.column_stack([XX.ravel(), YY.ravel()]), box, basis)
    return contour_generator(XX, YY, sdf.reshape(grid_n, grid_n)).lines(level)


def _curve_lipschitz(pts, vals, pct: float = 99.0, safety: float = 1.2) -> float:
    """Lipschitz constant of a function sampled along a curve, from neighbour-point slopes
    |dv|/|dp| (wraps around, assuming the points trace the loop in order)."""
    dp = np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1) + 1e-9
    dv = np.abs(np.diff(np.asarray(vals, float), append=np.asarray(vals, float)[:1]))
    return float(np.percentile(dv / dp, pct)) * safety + 1e-6


def encode_surface_function(pts, vals, box, *, grid_n: int = 64, order: int = 12,
                            basis: str = "cosine", L: float = None, mode: str = "mid"):
    """DCT code of a function sampled at interface points: McShane-Whitney extend (mid) to
    the box, then project.  `pts`: (N,2) interface points; `vals`: (N,) values.  Returns
    z_gamma (order^2,).  L defaults to a high-percentile neighbour slope."""
    pts = np.asarray(pts, float); vals = np.asarray(vals, float)
    if L is None:
        L = _curve_lipschitz(pts, vals)
    return extend_and_project(pts, vals, L, box, grid_n, order, basis, mode)


def decode_surface_function(z_gamma, query_pts, box, *, basis: str = "cosine"):
    """Evaluate the reconstructed surface function (truncated cosine series) at query points."""
    return basis_reconstruct(z_gamma, np.asarray(query_pts, float), box, basis)


def encode_state(loops, vals_per_loop, box, *, geom_order: int = 16, gam_order: int = 16,
                 grid_n: int = 64, basis: str = "cosine"):
    """Encode a full (interface, surface-function) state -> (z_geom, z_gamma).
    `loops`: list of (Ni,2) closed curves; `vals_per_loop`: list of (Ni,) values."""
    z_geom = encode_interface(loops, box, grid_n=grid_n, order=geom_order, basis=basis)
    pts = np.vstack([np.asarray(l, float) for l in loops])
    vals = np.concatenate([np.asarray(v, float) for v in vals_per_loop])
    z_gam = encode_surface_function(pts, vals, box, grid_n=grid_n, order=gam_order, basis=basis)
    return z_geom, z_gam


def decode_state(z_geom, z_gamma, box, *, grid_n: int = 240, basis: str = "cosine"):
    """Decode -> (loops, vals_per_loop): interface loops via marching squares of the decoded
    SDF, and the surface function evaluated at each loop's points."""
    loops = decode_interface(z_geom, box, grid_n=grid_n, basis=basis)
    vals = [decode_surface_function(z_gamma, lp, box, basis=basis) for lp in loops]
    return loops, vals
