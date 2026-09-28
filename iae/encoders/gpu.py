"""GPU (torch) batched encoders: IAE and the MFE moment baseline.

The per-sample encode (point-to-segment distance for the SDF, plus the
triangle-quadrature moment einsums) is the CPU bottleneck.  These batched torch
implementations pad a chunk of variable-size meshes to common dimensions (with
masks) and run the whole chunk on the GPU.  Results match the numpy encoders in
:mod:`iae.encoders.interface` to float precision.

    encode_batch(samples, encoder="iae"|"mfe", device="cuda", ...)
        -> dict(zgeom, zk, zf, zg)   each [B, .]

The two encoders differ only in ``zgeom``: "iae" projects the signed-distance
field of the boundary, "mfe" takes Legendre moments of the domain indicator.
The function codes ``zk, zf, zg`` are the same Legendre moments in both.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

_BARY = torch.tensor([[1 / 6, 1 / 6, 2 / 3], [1 / 6, 2 / 3, 1 / 6], [2 / 3, 1 / 6, 1 / 6]])
_TW = torch.tensor([1 / 3, 1 / 3, 1 / 3])
_gx, _gw = leggauss(5)
_EGT = torch.tensor(0.5 * (_gx + 1.0))   # 5-pt Gauss nodes on [0,1] for boundary edges
_EGW = torch.tensor(0.5 * _gw)


def _basis(u: torch.Tensor, order: int, kind: str) -> torch.Tensor:
    """Orthonormal-on-[0,1] basis evaluated at u; returns [..., order]."""
    if kind == "cosine":
        m = torch.arange(order, device=u.device, dtype=u.dtype)
        b = torch.cos(np.pi * u.unsqueeze(-1) * m)
        scale = torch.where(m == 0, torch.ones_like(m), torch.sqrt(torch.tensor(2.0, device=u.device)))
        return b * scale
    if kind == "legendre":
        x = 2 * u - 1
        cols = [torch.ones_like(x)]
        if order > 1:
            cols.append(x)
        for k in range(1, order - 1):
            cols.append(((2 * k + 1) * x * cols[-1] - k * cols[-2]) / (k + 1))
        P = torch.stack(cols[:order], dim=-1)
        norm = torch.sqrt(2 * torch.arange(order, device=u.device, dtype=u.dtype) + 1)
        return P * norm
    raise ValueError(kind)


def _pad(arrays, fill, dtype, device):
    """List of [n_i, ...] -> ([B, maxn, ...], mask [B, maxn])."""
    B = len(arrays)
    maxn = max(len(a) for a in arrays)
    tail = arrays[0].shape[1:]
    out = torch.full((B, maxn, *tail), fill, dtype=dtype, device=device)
    mask = torch.zeros((B, maxn), dtype=torch.bool, device=device)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = torch.as_tensor(a, dtype=dtype, device=device)
        mask[i, : len(a)] = True
    return out, mask


def _moments(points, fields, tris, tri_mask, basis, order):
    """Batched MFE-style domain moments. points[B,N,2], fields[B,F,N], tris[B,T,3] -> [B,F,order^2]."""
    B, T, _ = tris.shape
    F = fields.shape[1]
    bidx = torch.arange(B, device=points.device).view(B, 1, 1)
    verts = points[bidx, tris]  # [B,T,3,2]
    vec1, vec2 = verts[:, :, 1] - verts[:, :, 0], verts[:, :, 2] - verts[:, :, 0]
    areas = 0.5 * torch.abs(vec1[..., 0] * vec2[..., 1] - vec1[..., 1] * vec2[..., 0])  # [B,T]
    bary = _BARY.to(points)
    quad = torch.einsum("qc,btcd->btqd", bary, verts)  # [B,T,3,2]
    bx = _basis(quad[..., 0], order, basis)  # [B,T,3,order]
    by = _basis(quad[..., 1], order, basis)
    warea = areas.unsqueeze(-1) * _TW.to(points) * tri_mask.unsqueeze(-1)  # [B,T,3]
    bidx2 = torch.arange(B, device=points.device).view(B, 1, 1, 1)
    find = torch.arange(F, device=points.device).view(1, F, 1, 1)
    fvert = fields[bidx2, find, tris.unsqueeze(1)]  # [B,F,T,3] field at triangle vertices
    fq = torch.einsum("bftc,qc->bftq", fvert, bary)  # [B,F,T,3] field at quadrature points
    z = torch.einsum("btqi,btqj,bftq,btq->bfij", bx, by, fq, warea)  # [B,F,order,order]
    return z.reshape(B, F, -1)


def _boundary_moment(points, g, bedges, emask, order):
    """Batched boundary (edge) moments of g, matching interface.boundary_moment.

    points[B,N,2], g[B,N], bedges[B,E,2] (long), emask[B,E] -> [B,order^2]."""
    B = points.shape[0]
    bidx = torch.arange(B, device=points.device).view(B, 1)
    A = points[bidx, bedges[..., 0]]   # [B,E,2]
    Bp = points[bidx, bedges[..., 1]]
    gA, gB = g[bidx, bedges[..., 0]], g[bidx, bedges[..., 1]]  # [B,E]
    length = torch.linalg.norm(Bp - A, dim=-1)  # [B,E]
    t = _EGT.to(points)  # [Q]
    quad = (1 - t).view(1, 1, -1, 1) * A.unsqueeze(2) + t.view(1, 1, -1, 1) * Bp.unsqueeze(2)  # [B,E,Q,2]
    gq = (1 - t).view(1, 1, -1) * gA.unsqueeze(2) + t.view(1, 1, -1) * gB.unsqueeze(2)  # [B,E,Q]
    wedge = _EGW.to(points).view(1, 1, -1) * length.unsqueeze(-1) * emask.unsqueeze(-1)  # [B,E,Q]
    lx = _basis(quad[..., 0], order, "legendre")
    ly = _basis(quad[..., 1], order, "legendre")
    z = torch.einsum("beqi,beqj,beq,beq->bij", lx, ly, gq, wedge)
    return z.reshape(B, -1)


def _signed_distance(query, qmask, A, Bp, emask, chunk_e=None):
    """Batched signed distance. query[B,N,2], A,Bp[B,E,2] edge ends, emask[B,E]."""
    # distances via batched matmul (bmm) -- no [B,N,E,2] temporary
    AB = Bp - A  # [B,E,2]
    AB2 = (AB * AB).sum(-1).clamp_min(1e-30)  # [B,E]
    A_AB = (A * AB).sum(-1)  # [B,E]
    sqA = (A * A).sum(-1)  # [B,E]
    sqQ = (query * query).sum(-1)  # [B,N]
    AQ_AB = torch.bmm(query, AB.transpose(1, 2)) - A_AB.unsqueeze(1)  # [B,N,E]
    AQ2 = sqQ.unsqueeze(-1) - 2.0 * torch.bmm(query, A.transpose(1, 2)) + sqA.unsqueeze(1)  # [B,N,E]
    t = (AQ_AB / AB2.unsqueeze(1)).clamp(0, 1)  # [B,N,E]
    d2 = AQ2 - 2.0 * t * AQ_AB + (t * t) * AB2.unsqueeze(1)
    d = torch.sqrt(d2.clamp_min(0.0)).masked_fill(~emask.unsqueeze(1), 1e30).amin(-1)  # [B,N]
    # even-odd ray casting (inside test) over edges
    x, y = query[..., 0:1], query[..., 1:2]  # [B,N,1]
    x1, y1 = A[..., 0].unsqueeze(1), A[..., 1].unsqueeze(1)  # [B,1,E]
    x2, y2 = Bp[..., 0].unsqueeze(1), Bp[..., 1].unsqueeze(1)
    straddle = (y1 > y) != (y2 > y)
    xc = (x2 - x1) * (y - y1) / (y2 - y1).where((y2 - y1) != 0, torch.ones_like(y2)) + x1
    cross = straddle & (x < xc) & emask.unsqueeze(1)
    inside = (cross.sum(-1) % 2) == 1  # [B,N]
    return torch.where(inside, -d, d)


def encode_batch(samples, encoder="iae", device="cuda", geom_basis="cosine",
                 geom_order=12, geom_mode="auto", geom_grid=24, mfe_mode=12, chunk=256):
    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    out = {k: [] for k in ("zgeom", "zk", "zf", "zg")}
    for s0 in range(0, len(samples), chunk):
        batch = samples[s0 : s0 + chunk]
        pts, pmask = _pad([s["points"] for s in batch], 0.0, torch.float64, dev)
        tris, tmask = _pad([s["triangles"] for s in batch], 0, torch.long, dev)

        bedges, bemask = _pad([s["bedges"] for s in batch], 0, torch.long, dev)
        # interior fields k,f: MFE triangle moments; boundary g: MFE 1-D edge moments
        kf = torch.zeros((len(batch), 2, pts.shape[1]), dtype=torch.float64, device=dev)
        gnodal = torch.zeros((len(batch), pts.shape[1]), dtype=torch.float64, device=dev)
        for i, s in enumerate(batch):
            n = len(s["points"])
            kf[i, 0, :n] = torch.as_tensor(s["k"], dtype=torch.float64, device=dev)
            kf[i, 1, :n] = torch.as_tensor(s["f"], dtype=torch.float64, device=dev)
            gnodal[i, :n] = torch.as_tensor(s["g"], dtype=torch.float64, device=dev)
        zkf = _moments(pts, kf, tris, tmask, "legendre", mfe_mode)  # [B,2,mode^2]
        zk, zfk = zkf[:, 0], zkf[:, 1]
        zg = _boundary_moment(pts, gnodal, bedges, bemask, mfe_mode)

        if encoder == "mfe":
            zgeom = _moments(pts, pmask.to(torch.float64).unsqueeze(1), tris, tmask, "legendre", mfe_mode)[:, 0]
        elif encoder == "iae":  # geometry = SDF coefficients in the chosen basis / rule
            B = len(batch)
            bi = torch.arange(B, device=dev).view(-1, 1)
            A, Bp = pts[bi, bedges[..., 0]], pts[bi, bedges[..., 1]]
            mode = ("dct" if geom_basis == "cosine" else "moment") if geom_mode == "auto" else geom_mode
            if mode == "dct":
                a = (torch.arange(geom_grid, device=dev, dtype=torch.float64) + 0.5) / geom_grid
                gx, gy = torch.meshgrid(a, a, indexing="ij")
                grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).unsqueeze(0).expand(B, -1, -1)
                gmask = torch.ones((B, grid.shape[1]), dtype=torch.bool, device=dev)
                sdf = _signed_distance(grid, gmask, A, Bp, bemask).reshape(B, geom_grid, geom_grid)
                bb = _basis(a, geom_order, geom_basis)  # [grid, order]
                zgeom = torch.einsum("bxy,xi,yj->bij", sdf, bb, bb).reshape(B, -1) * (1.0 / geom_grid) ** 2
            elif mode == "moment":
                sdf = _signed_distance(pts, pmask, A, Bp, bemask) * pmask  # [B,N]
                zgeom = _moments(pts, sdf.unsqueeze(1), tris, tmask, geom_basis, geom_order)[:, 0]
            else:
                raise ValueError(f"unknown geom_mode {geom_mode}")
        else:
            raise ValueError(f"unknown encoder {encoder}")
        for key, val in (("zgeom", zgeom), ("zk", zk), ("zf", zfk), ("zg", zg)):
            out[key].append(val.cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
