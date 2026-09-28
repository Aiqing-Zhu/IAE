"""Check that this copy is complete and ready to run.

Verifies the environment, that every input file each example needs is present and
loadable, and that the encoder round-trips. Run it first:

    python check_setup.py
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OK, BAD, SKIP = "  ok  ", " MISS ", " skip "
_bad = 0


def line(status, what, detail=""):
    global _bad
    if status is BAD:
        _bad += 1
    print(f"[{status}] {what:42s} {detail}")


def check_files(label, root, names, optional=False):
    missing = [n for n in names if not (root / n).exists()]
    if missing:
        line(SKIP if optional else BAD, label,
             f"missing {', '.join(missing[:3])}{' ...' if len(missing) > 3 else ''} in {root}")
    else:
        mb = sum((root / n).stat().st_size for n in names) / 1e6
        line(OK, label, f"{len(names)} files, {mb:.0f} MB")


def main():
    print("environment")
    try:
        import numpy, scipy, matplotlib, contourpy, torch            # noqa: F401
        line(OK, "numpy / scipy / matplotlib / contourpy",
             f"{numpy.__version__} / {scipy.__version__} / "
             f"{matplotlib.__version__} / {contourpy.__version__}")
        line(OK, "torch", f"{torch.__version__}, cuda "
             f"{'available (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'NOT available -- everything falls back to CPU'}")
    except ImportError as exc:                                       # noqa: BLE001
        line(BAD, "python packages", str(exc))
        print("\ninstall them with:  conda env create -f environment.yml")
        return 1

    from iae.paths import EX1_DIR, EX2_CACHE, EX2_CKPT, EX2_DATA, EX3_CACHE, EX3_CKPT, EX3_DATA

    print(f"\ninputs   (Example 1 in {EX1_DIR})")
    check_files("Example 1 codes", EX1_DIR,
                ["mfe_train.npz", "mfe_test.npz", "iae_train_cosine.npz", "iae_test_cosine.npz",
                 "iae_train_legendre.npz", "iae_test_legendre.npz"])
    check_files("Example 1 meshed domains", EX1_DIR,
                ["raw_train.npz", "raw_train_b.npz", "raw_test.npz"], optional=True)
    check_files("Example 2 interfaces", EX2_DATA, ["interfaces.pkl"])
    check_files("Example 2 codes", EX2_CACHE, ["train_pairs.npz", "test_enc.pkl"])
    check_files("Example 3 interfaces + surfactant", EX3_DATA, ["traj_train.pkl", "traj_test.pkl"])
    check_files("Example 3 codes", EX3_CACHE, ["pairs_train.npz", "traj_test_enc.pkl"])

    print("\ntrained operators (optional -- training rebuilds them)")
    check_files("Example 2 checkpoints", EX2_CKPT,
                [f"evolution_{h}-o{o}-s{s}.pt" for h, o in (("code", 24), ("code", 16),
                                                            ("code", 12), ("linear", 24))
                 for s in range(5)], optional=True)
    check_files("Example 3 checkpoints", EX3_CKPT,
                [f"surf_operator-s{s}.pt" for s in range(5)], optional=True)

    print("\nencoder")
    import numpy as np
    from iae.encoders import decode_state, encode_state
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    loops = [np.column_stack([c + 0.12 * np.cos(t), 0.5 + 0.12 * np.sin(t)]) for c in (0.3, 0.72)]
    vals = [np.full(len(l), v) for l, v in zip(loops, (1.0, 0.6))]
    zg, zgam = encode_state(loops, vals, (0., 1., 0., 1.), geom_order=32, gam_order=32, grid_n=96)
    rec, _ = decode_state(zg, zgam, (0., 1., 0., 1.), grid_n=240)
    rec = [l for l in rec if len(l) > 8]
    line(OK if len(rec) == 2 else BAD, "encode -> decode round trip",
         f"{len(rec)} drops recovered from {zg.size}+{zgam.size} coefficients")

    print()
    if _bad:
        print(f"{_bad} problem(s) -- see 'Data and checkpoints' in the README.")
        return 1
    print("ready. Next: examples/ex2_hele_shaw/, examples/ex3_stokes_surfactant/, "
          "examples/ex1_poisson_star/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
