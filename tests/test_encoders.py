"""Self-contained tests for the IAE encoders (no dataset required).

Run either way:

    python tests/test_encoders.py
    pytest tests/test_encoders.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from iae.encoders import (basis_1d, decode_interface, decode_state, decode_surface_function,
                          encode_interface, encode_state, encode_surface_function,
                          mcshane_extend, signed_distance_from_edges)
from iae.encoders.interface import _loops_to_edges

BOX = (0.0, 1.0, 0.0, 1.0)


def circle(c, r, n=400):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def hausdorff(A, B):
    d = np.sqrt(((np.asarray(A)[:, None] - np.asarray(B)[None]) ** 2).sum(-1))
    return max(d.min(1).max(), d.min(0).max())


def test_basis_orthonormal():
    """Both bases are orthonormal in L2([0,1]) as functions."""
    x, w = np.polynomial.legendre.leggauss(64)
    a, wa = 0.5 * (x + 1.0), 0.5 * w
    for kind in ("cosine", "legendre"):
        B = basis_1d(a, 8, kind)
        gram = B.T @ (wa[:, None] * B)
        assert np.abs(gram - np.eye(8)).max() < 1e-10, f"{kind} basis not orthonormal"


def test_cosine_is_orthonormal_on_the_encoder_grid():
    """The property the encoder actually relies on: on the midpoint grid the cosine basis is
    DISCRETELY orthonormal (it is a DCT-II), so the midpoint-rule projection in `_project_grid`
    is exact and `basis_reconstruct` inverts it.  Legendre is orthonormal only in the continuum:
    on the same grid its discrete Gram is far from the identity, which is why nothing in this
    repository decodes a Legendre grid-projected code."""
    for order, grid_n in ((12, 24), (24, 48), (32, 96)):
        a = (np.arange(grid_n) + 0.5) / grid_n
        cos_err = np.abs((lambda B: B.T @ B / grid_n)(basis_1d(a, order, "cosine")) - np.eye(order)).max()
        leg_err = np.abs((lambda B: B.T @ B / grid_n)(basis_1d(a, order, "legendre")) - np.eye(order)).max()
        assert cos_err < 1e-12, f"cosine not discretely orthonormal at order {order}, grid {grid_n}"
        assert leg_err > 1e-3, ("legendre unexpectedly discretely orthonormal -- if this now holds, "
                                "the decode-side restriction documented in extension.py can be lifted")


def test_mcshane_interpolates_and_is_lipschitz():
    """The mid extension reproduces the data and does not exceed the Lipschitz bound."""
    rng = np.random.default_rng(0)
    pts = circle((0.5, 0.5), 0.3, 120)
    vals = np.sin(4 * np.arctan2(pts[:, 1] - 0.5, pts[:, 0] - 0.5))
    L = 20.0                                    # > 4 / 0.3, the data's own Lipschitz constant
    assert np.abs(mcshane_extend(pts, pts, vals, L) - vals).max() < 1e-10, \
        "the mid extension must interpolate the data"

    q = rng.random((200, 2))
    fq = mcshane_extend(q, pts, vals, L)
    d = np.linalg.norm(q[:, None] - q[None], axis=-1) + np.eye(len(q))
    slope = np.abs(fq[:, None] - fq[None]) / d
    assert slope.max() <= L + 1e-8, f"extension slope {slope.max():.3f} exceeds L={L}"


def test_signed_distance_matches_analytic_circle():
    """SDF from the edge set agrees with the exact circle distance away from the polygon."""
    loop = circle((0.5, 0.5), 0.25, 2000)
    pts, edges = _loops_to_edges([loop])
    rng = np.random.default_rng(1)
    q = rng.random((500, 2))
    got = signed_distance_from_edges(q, pts, edges)
    exact = np.linalg.norm(q - 0.5, axis=1) - 0.25
    assert np.abs(got - exact).max() < 2e-4


def test_interface_round_trip_recovers_geometry_and_topology():
    """Two well-separated drops encode and decode back to two drops."""
    loops = [circle((0.3, 0.5), 0.12), circle((0.72, 0.5), 0.10)]
    z = encode_interface(loops, BOX, grid_n=96, order=32)
    dec = [np.asarray(l) for l in decode_interface(z, BOX, grid_n=240) if len(l) > 8]
    assert len(dec) == 2, f"topology lost: {len(dec)} loops decoded"
    err = max(min(hausdorff(d, r) for r in loops) for d in dec)
    assert err < 0.01, f"interface Hausdorff {err:.4f} too large"


def test_surface_function_round_trip():
    """A smooth function on the interface survives encode -> decode."""
    loop = circle((0.5, 0.5), 0.25, 600)
    theta = np.arctan2(loop[:, 1] - 0.5, loop[:, 0] - 0.5)
    vals = 1.0 + 0.3 * np.cos(2 * theta)
    z = encode_surface_function(loop, vals, BOX, grid_n=96, order=32)
    got = decode_surface_function(z, loop, BOX)
    rel = np.linalg.norm(got - vals) / np.linalg.norm(vals)
    assert rel < 0.05, f"surface function relative error {rel:.4f}"


def test_state_round_trip():
    """encode_state / decode_state keep both the drops and the values on them."""
    loops = [circle((0.32, 0.5), 0.13), circle((0.72, 0.52), 0.11)]
    vals = [1.0 + 0.2 * np.cos(3 * np.arctan2(l[:, 1] - l[:, 1].mean(), l[:, 0] - l[:, 0].mean()))
            for l in loops]
    zg, zgam = encode_state(loops, vals, BOX, geom_order=32, gam_order=32, grid_n=96)
    dec_loops, dec_vals = decode_state(zg, zgam, BOX, grid_n=240)
    dec = [(l, v) for l, v in zip(dec_loops, dec_vals) if len(l) > 8]
    assert len(dec) == 2
    for l, v in dec:
        j = int(np.argmin([hausdorff(l, r) for r in loops]))
        assert hausdorff(l, loops[j]) < 0.01
        ref = np.interp(np.arange(len(v)), np.arange(len(vals[j])), vals[j])
        assert np.abs(v.mean() - ref.mean()) < 0.1


def test_gpu_encoder_matches_numpy():
    """iae.encoders.gpu reproduces the numpy geometry code (run on the CPU device)."""
    import torch                                                        # noqa: F401
    from iae.encoders import gpu, sdf_geometry_code

    loop = circle((0.5, 0.5), 0.3, 200)
    pts, edges = _loops_to_edges([loop])
    tri = np.array([[0, i, i + 1] for i in range(1, len(loop) - 1)])     # fan triangulation
    s = dict(points=pts, triangles=tri, bedges=edges,
             k=np.ones(len(pts)), f=np.ones(len(pts)), g=np.ones(len(pts)))
    z_gpu = gpu.encode_batch([s], encoder="iae", device="cpu", geom_basis="cosine",
                             geom_order=12, geom_mode="dct", geom_grid=24)["zgeom"][0]
    z_np = sdf_geometry_code(pts, tri, edges, "cosine", 12, "dct", 24)
    assert np.abs(z_gpu - z_np).max() < 1e-5


def test_mionet_forward_shapes():
    from iae.models import MIONet
    import torch

    net = MIONet([[9, 16, 8], [9, 16, 8], [2, 16, 8]], "relu", bias=True)
    out = net([torch.zeros(4, 9), torch.zeros(4, 9), torch.zeros(4, 11, 2)])
    assert out.shape == (4, 11)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:                                        # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
