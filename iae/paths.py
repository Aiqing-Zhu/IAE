"""Where the datasets, the encodings and the checkpoints live.

Every example resolves its files through this module.

    <repo>/data/ex2_hele_shaw/           interfaces.pkl
    <repo>/data/ex3_stokes_surfactant/   traj_train.pkl  traj_test.pkl
    <repo>/cache/ex2_hele_shaw/          train_pairs.npz  test_enc.pkl
    <repo>/cache/ex3_stokes_surfactant/  pairs_train.npz  traj_test_enc.pkl
    <repo>/checkpoints/ex{2,3}_*/        trained operators
    ../poisson_data/                     everything for Example 1 (see below)

Examples 2 and 3 are self-contained inside the repository: what they need is the
interface (and, for Example 3, the surfactant on it) plus the codes, and all of it
fits in a few hundred MB.

**Example 1 is different.**  Its meshed domains are 2.2 GB and its codes another
450 MB, so the whole example is distributed separately and lives in one flat
directory *beside* the repository, `../poisson_data` by default:

    raw_train.npz  raw_train_b.npz  raw_test.npz        the meshed domains
    mfe_{train,test}.npz                                query points, targets, MFE codes
    iae_{train,test}_{cosine,legendre}.npz              IAE codes, both bases

Training reads only the codes; the raw files are needed to rebuild them
(`encode_mfe.py`, `encode_iae.py`) and by the GINO baseline.

Every location can be overridden:

    IAE_DATA_ROOT    interfaces / trajectories    (default: <repo>/data)
    IAE_CACHE_ROOT   codes for Examples 2 and 3   (default: <repo>/cache)
    IAE_EX1_DIR      everything for Example 1     (default: <repo>/../poisson_data)
    IAE_CKPT_ROOT    trained operators            (default: <repo>/checkpoints)
"""

from __future__ import annotations

import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

DATA_ROOT = pathlib.Path(os.environ.get("IAE_DATA_ROOT", REPO / "data")).expanduser()
CACHE_ROOT = pathlib.Path(os.environ.get("IAE_CACHE_ROOT", REPO / "cache")).expanduser()
CKPT_ROOT = pathlib.Path(os.environ.get("IAE_CKPT_ROOT", REPO / "checkpoints")).expanduser()

_EXAMPLES = ("ex2_hele_shaw", "ex3_stokes_surfactant")

EX2_DATA, EX3_DATA = (DATA_ROOT / e for e in _EXAMPLES)
EX2_CACHE, EX3_CACHE = (CACHE_ROOT / e for e in _EXAMPLES)
EX2_CKPT, EX3_CKPT = (CKPT_ROOT / e for e in _EXAMPLES)

# Example 1 is distributed as one directory next to the repository, holding both the
# meshed domains and the codes built from them.
EX1_DIR = pathlib.Path(os.environ.get("IAE_EX1_DIR", REPO.parent / "poisson_data")).expanduser()
EX1_DATA = EX1_CACHE = EX1_DIR


def need(path: pathlib.Path) -> pathlib.Path:
    """Return `path`, or fail with a message that says what is missing and why."""
    if not path.exists():
        raise SystemExit(
            f"missing input: {path}\n"
            f"  data   = {DATA_ROOT}\n"
            f"  cache  = {CACHE_ROOT}\n"
            f"  ex1    = {EX1_DIR}\n"
            f"  ckpt   = {CKPT_ROOT}\n"
            "Override with IAE_DATA_ROOT / IAE_CACHE_ROOT / IAE_EX1_DIR / IAE_CKPT_ROOT,\n"
            "or see 'Data and checkpoints' in the README."
        )
    return path


def out_dir(path: pathlib.Path) -> pathlib.Path:
    """Return an output directory (codes / checkpoints), creating it if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path
