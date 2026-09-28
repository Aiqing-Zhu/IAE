"""Build the distributable data bundles for Examples 2 and 3.

Each bundle is a gzipped tar whose members carry repository-relative paths, so it is
unpacked with

    tar xzf iae-data-ex2_hele_shaw.tar.gz -C /path/to/this/repo

What goes in, and why, is decided by what the operators actually read:

  ex2_hele_shaw          interfaces.pkl    the interface polylines of every trajectory
                         train_pairs.npz   codes, SDF targets, auxiliary pressure samples
                         test_enc.pkl      test codes + the reference interfaces
      The finite-element meshes and pressure fields (4.5 GB) are deliberately left out:
      nothing downstream reads them once the codes exist, and `train_evolution.py
      reencode` rebuilds the cache from interfaces.pkl at any grid or order.

  ex3_stokes_surfactant  traj_{train,test}.pkl   interfaces AND the surfactant on them
                         pairs_train.npz         training codes
                         traj_test_enc.pkl       test codes
      Here the raw data already is the interface + surface function, so it ships as is.

Symlinked source directories are dereferenced, so the bundles are self-contained.

  python tools/package_data.py [outdir=dist]
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from iae.paths import CACHE_ROOT, DATA_ROOT, need, out_dir               # noqa: E402

BUNDLES = {
    "ex2_hele_shaw": [
        (DATA_ROOT, "data", ["interfaces.pkl"]),
        (CACHE_ROOT, "cache", ["train_pairs.npz", "test_enc.pkl"]),
    ],
    "ex3_stokes_surfactant": [
        (DATA_ROOT, "data", ["traj_train.pkl", "traj_test.pkl"]),
        (CACHE_ROOT, "cache", ["pairs_train.npz", "traj_test_enc.pkl"]),
    ],
}


def sha256_file(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = out_dir(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "dist"))
    lines = []
    for example, groups in BUNDLES.items():
        tar_path = out / f"iae-data-{example}.tar.gz"
        total = 0
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, prefix, names in groups:
                for name in names:
                    src = need((root / example / name).resolve())   # resolve: deref symlinks
                    arc = f"{prefix}/{example}/{name}"
                    total += src.stat().st_size
                    print(f"  + {arc}  ({src.stat().st_size/1e6:.0f} MB)", flush=True)
                    tar.add(src, arcname=arc)
        digest = sha256_file(tar_path)
        lines.append(f"{digest}  {tar_path.name}")
        print(f"{tar_path.name}: {tar_path.stat().st_size/1e6:.0f} MB "
              f"(from {total/1e6:.0f} MB of files)\n", flush=True)
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print("SHA256SUMS:")
    print("\n".join("  " + x for x in lines))
    print(f"\nunpack with:  tar xzf <bundle>.tar.gz -C {pathlib.Path(__file__).resolve().parents[1]}")


if __name__ == "__main__":
    main()
