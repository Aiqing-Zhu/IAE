# Interface Autoencoder (IAE)

Reference implementation for *Learning Operators of Geometry with an Interface
Autoencoder*.

An **interface autoencoder** is a finite-dimensional representation of a *state*
`(Γ, f)` — a curve `Γ` in the plane together with a function `f` defined on it. It
is built from two primitives:

* **extend** — a function known only on `Γ` is extended to the whole bounding box by
  the McShane–Whitney (Lipschitz) mid-extension; a *region* is extended by the same
  construction, its signed-distance field being the one-sided `(g = 0, L = 1)`
  extension of its boundary, signed by the inside/outside test;
* **project** — the extended field is projected onto a truncated orthonormal tensor
  basis, giving codes `z_geom` and `z_γ`.

Decoding is resolution-free: evaluate the truncated series anywhere. The interface
comes back as the **zero level set** of the decoded signed-distance field, so the
number of connected components is an *output* of the decoder rather than an input.
That is what lets an operator trained in code space follow a topology change — two
drops merging into one — without any special handling.

Operators are then learned entirely in code space: a MIONet maps the initial codes
(plus a discrete time index, where the problem is an evolution) to the codes of the
state at a later time.

```
(Γ, f) --encode--> (z_geom, z_γ) --MIONet--> (ẑ_geom, ẑ_γ) --decode--> (Γ̂, f̂)
```

---

## Repository layout

```
iae/                     the library
  encoders/
    extension.py         McShane-Whitney extension, orthonormal bases, projection,
                         signed distance to an edge set
    interface.py         the autoencoder: encode/decode a state, an interface,
                         a function on an interface; the Example-1 encoder
    gpu.py               batched torch version, for encoding whole datasets
    utils.py             quadrature constants, polygon -> SDF
  models/onet.py         MIONet (product of branch nets x a trunk net)
  paths.py               where the data, codes and checkpoints are looked up
check_setup.py           one-command check that this copy is complete
environment.yml          the exact versions everything was verified with
examples/
  ex1_poisson_star/      variable-domain Poisson: encode geometry, learn (Ω,k,f,g) -> u
  ex2_hele_shaw/         Hele-Shaw: interface evolution through merges
  ex3_stokes_surfactant/ Stokes flow: joint interface + surfactant evolution
tools/package_data.py    builds the distributable Example-2 / Example-3 bundles
tests/                   self-contained encoder tests, no dataset needed
```

Data generation is **not** part of this repository: the examples read pre-generated
datasets. The finite-element and level-set solvers that produced them are described
in the appendices of the paper.

---

## Installation

```bash
conda env create -f environment.yml    # the versions this was verified with
conda activate iae
```

or, into an environment you already have, `pip install -r requirements.txt`
(numpy, scipy, torch, matplotlib, contourpy — looser bounds).

Optionally `pip install -e .` to import `iae` from anywhere; the example scripts
add the repository root to `sys.path` themselves, so it is not required.

A GPU is used for encoding whole datasets and for training. Everything falls back
to the CPU if `torch.cuda.is_available()` is false, at the obvious cost.

---

## Data and checkpoints

For Examples 2 and 3, download the
[data and checkpoints archive](https://drive.google.com/file/d/1BeBehr6ZumflVxgqJW7HRgjXJx2Kr9tz/view?usp=sharing) (about 348 MB)
as `IAE_2_3_data.zip` into the repository root and extract it there:

```bash
python -m zipfile -e IAE_2_3_data.zip .
```

The archive contains top-level `data/`, `cache/`, and `checkpoints/` directories.
Example 1 uses a separate 2.6 GB
[Poisson data archive](https://drive.google.com/file/d/1iGOd-tSLC4QZ2drSR5TVX9nn1sIs19vV/view?usp=sharing)
with a top-level `poisson_data/` directory. From the repository root, save it as
`../poisson_data.zip` and extract it into the parent directory:

```bash
python -m zipfile -e ../poisson_data.zip ..
```

The resulting layout is:

```
poisson_data/                         <- all of Example 1 (downloaded separately, 2.6 GB)
  raw_{train,train_b,test}.npz           the meshed domains
  mfe_{train,test}.npz                   query points, targets, 2D/1D-MFE codes
  iae_{train,test}_{cosine,legendre}.npz IAE codes, both bases
this folder/                          <- repository root
  data/ex2_hele_shaw/interfaces.pkl        the interfaces of all 1387 trajectories
  data/ex3_stokes_surfactant/traj_*.pkl    interfaces AND the surfactant on them
  cache/ex2_hele_shaw/                     train_pairs.npz  test_enc.pkl
  cache/ex3_stokes_surfactant/             pairs_train.npz  traj_test_enc.pkl
  checkpoints/ex{2,3}_*/                   trained operators, 5 seeds each
```

So `poisson_data/` and the repository root must be siblings. Every location can be
moved with an environment variable:

| variable | default | contents |
|---|---|---|
| `IAE_EX1_DIR` | `../poisson_data` | all of Example 1 |
| `IAE_DATA_ROOT` | `data/` | interfaces and trajectories |
| `IAE_CACHE_ROOT` | `cache/` | Example-2/3 codes |
| `IAE_CKPT_ROOT` | `checkpoints/` | trained operators |

---

## Quick start

```bash
python check_setup.py                  # is everything here and importable?  (seconds)
python tests/test_encoders.py          # 9 encoder tests, no data needed
```

`check_setup.py` reports the package versions, whether a GPU is visible, and which
input files each example can see — run it first.

The autoencoder on its own, without any dataset:

```python
import numpy as np
from iae.encoders import encode_state, decode_state

def circle(c, r, n=400):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])

box   = (0.0, 1.0, 0.0, 1.0)
loops = [circle((0.3, 0.5), 0.12), circle((0.72, 0.5), 0.10)]     # two drops
vals  = [np.full(len(l), v) for l, v in zip(loops, (1.0, 0.6))]   # a function on them

z_geom, z_gamma = encode_state(loops, vals, box, geom_order=32, gam_order=32, grid_n=96)
rec_loops, rec_vals = decode_state(z_geom, z_gamma, box, grid_n=240)
print(len(rec_loops), "drops recovered from", z_geom.shape, "+", z_gamma.shape, "coefficients")
```

---

## Reproducing the paper

Each example is three steps: **encode once → train → evaluate**. Set `GPUS` to the
GPUs you want the run scripts to use, e.g. `GPUS="0 1 3"`.

### Example 1 — variable-domain Poisson (Table 1)

A four-branch MIONet `(geometry, k, f, g) × trunk(x)` predicts `u`. The four
encoders differ only in the geometry code (and, for IAE, in how `g` is coded);
`k, f` are Legendre domain moments throughout.

The codes are in `../poisson_data`, so training starts immediately:

```bash
cd examples/ex1_poisson_star
GPUS="0 1" bash run_ex1.sh                 # 4 encoders x 3 normalizations x 5 seeds = 60 runs
python aggregate_ex1.py                    # -> Table 1
```

To rebuild the codes from the meshed domains instead (they ship too):

```bash
python encode_mfe.py                       # query points + the MFE baseline codes   (4 min, GPU)
python encode_iae.py 32 cosine             # IAE codes, cosine basis                 (1 min, 32 cores)
python encode_iae.py 32 legendre           # IAE codes, Legendre basis
```

One run is 120000 updates at batch 64 (about 40 minutes on an L40S). A single
configuration can be run directly:

```bash
python train_ex1.py iae_cosine lowfreq 0   # [encoder] [normalization] [seed] [iters]
```

Encoders: `iae_cosine`, `iae_legendre`, `mfe2d`, `mfe1d`. Normalizations: `none`,
`perdim`, `lowfreq`. Reported errors are relative L2 at every mesh node of the test
domains, at the **final** checkpoint — no checkpoint is selected on the test set.

Reference (paper Table 1, mean ± std over 5 seeds, `overall / notched / smooth`):

| encoder | normalization | overall | notched | smooth |
|---|---|---|---|---|
| 2D-MFE | per-dim | 0.0858 ± 0.0013 | 0.1135 ± 0.0016 | 0.0593 ± 0.0013 |
| 1D-MFE | per-dim | 0.0653 ± 0.0011 | 0.0826 ± 0.0013 | 0.0487 ± 0.0010 |
| IAE (cosine) | low-freq | 0.0507 ± 0.0005 | 0.0607 ± 0.0006 | 0.0412 ± 0.0005 |
| IAE (Legendre) | low-freq | 0.0505 ± 0.0003 | 0.0603 ± 0.0002 | 0.0411 ± 0.0005 |
| GINO | — | 0.0773 ± 0.0017 | 0.0852 ± 0.0022 | 0.0697 ± 0.0011 |

### Example 2 — Hele-Shaw interface evolution (Table 2, Figure 1)

A time-conditioned operator maps `(Γ(0), source)` and a discrete time `t_k = 0.56k`
**directly** to `Γ(t_k)` — no autoregression, no re-initialization, so a merge is
predicted in one shot from `t = 0`.

The Example 2 download contains precomputed codes, so training can start after
extraction:

```bash
cd examples/ex2_hele_shaw
GPUS="0 1" bash run_ex2.sh                 # 4 configurations x 5 seeds, then table + figure
```

or individually:

```bash
python train_evolution.py 40000 code 24 0  # [iters] [head: code|linear] [order] [seed]
python eval_evolution.py table             # 5-seed stratified d_H
python eval_evolution.py figure            # figures/ex2_rollout.png
```

To rebuild the cache rather than use the shipped one:

```bash
python train_evolution.py reencode 48 24   # rebuild the default cache from interfaces.pkl
python train_evolution.py encode           # from the meshes (4.5 GB, not distributed)
python extract_interfaces.py               # meshes -> interfaces.pkl (how the bundle is built)
```

The training code currently assumes a 48-point grid and a stored code order of 24,
so use those values when rebuilding its cache.

`reencode` reproduces the mesh-built codes to `7e-9` and carries the source codes and
the auxiliary pressure samples over unchanged, since neither depends on the geometry
discretization.

Reference (paper Table 2, two-sided Hausdorff over all predicted frames):

| configuration | overall `d_H` | merge | non-merge |
|---|---|---|---|
| nonlinear, r = 24 | 0.0110 ± 0.0001 | 0.0131 ± 0.0001 | 0.0081 ± 0.0001 |
| nonlinear, r = 16 | 0.0129 ± 0.0000 | 0.0162 ± 0.0001 | 0.0083 ± 0.0000 |
| nonlinear, r = 12 | 0.0142 ± 0.0001 | 0.0184 ± 0.0001 | 0.0087 ± 0.0000 |
| linear, r = 24 | 0.0119 ± 0.0001 | 0.0149 ± 0.0001 | 0.0080 ± 0.0001 |

`r` is the truncation order of the predicted code; the order-24 code is stored once
and cut down to `r × r`, which is exact for an orthonormal basis.

### Example 3 — Stokes flow with surfactant (Table 3, Figure 2)

The same idea with **two** code heads: interface *and* the surfactant living on it
are predicted jointly from the initial state.

The Example 3 download contains both the interface + surfactant trajectories and their
codes, so training can start after extraction:

```bash
cd examples/ex3_stokes_surfactant
GPUS="0 1" bash run_ex3.sh                 # 5 seeds, then table + gallery figure
python train_operator.py encode            # (only to rebuild the codes yourself, 29 min)
```

or individually:

```bash
python train_operator.py 30000 0           # [iters] [seed]
python eval_operator.py                    # -> Table 3
python plot_gallery.py                     # figures/ex3_gallery.png
python diag_adjacency.py                   # appendix: concatenated vs per-loop Lipschitz adjacency
```

Reference (paper Table 3, 150 test trajectories / 1158 evaluated frames):

| regime | # test | interface `d_H` | state `d_gr` | loop count |
|---|---|---|---|---|
| shearing-past | 96 | 0.0042 ± 0.0000 | 0.0258 ± 0.0004 | 100% |
| isolated | 32 | 0.0049 ± 0.0001 | 0.0413 ± 0.0010 | 100% |
| co-translating | 22 | 0.0057 ± 0.0003 | 0.0367 ± 0.0010 | 100% |
| **overall** | **150** | **0.0046 ± 0.0001** | **0.0305 ± 0.0003** | **100%** |

`d_H` is the two-sided Hausdorff distance between the union of all predicted loops
and the union of all reference loops after mapping the physical box to `[0,1]²`;
`d_gr` is the two-sided Hausdorff distance between the state graphs `{(x, f(x))}`
under `d((x,s),(y,t)) = max(‖x−y‖, |s−t|)`. Neither uses per-loop matching.


---

## Library reference

```python
from iae.encoders import (
    encode_state, decode_state,                        # a full (interface, function) state
    encode_interface, decode_interface,                # geometry only; decode returns loops
    encode_surface_function, decode_surface_function,  # a function sampled on interface points
    encode_gext,                                       # Example-1: (Ω, k, f, g) -> four codes
    mcshane_extend, basis_1d, extend_and_project, basis_reconstruct,
    signed_distance_from_edges,
)
from iae.encoders import gpu                           # gpu.encode_batch(samples, encoder="iae"|"mfe")
from iae.models import MIONet, FNN
```

Two hyperparameters govern the representation: the **basis** and the **order** `r`,
giving `r²` coefficients per field; `grid_n` is the projection grid.

**On the choice of basis.** Everything that *decodes* here uses the cosine basis,
for which the grid projection is a DCT-II: it is discretely orthonormal on the
projection grid, so `basis_reconstruct` exactly inverts the projection and
`grid_n ≥ 2r` suffices. The Legendre basis is orthonormal as a *function* basis but
not on that grid, so Legendre grid coefficients are sound as **features** — which is
how Example 1 uses them, comparing them against the cosine ones as encoders — while
reconstructing a field from them needs a far finer grid than `2r`. The Example-1
`k, f` and `g` moments are a different, exact construction (Gauss quadrature on the
mesh) and are unaffected.

---

## Notes

* Results are reported as mean ± **population** standard deviation (`ddof = 0`)
  over five seeds, matching the paper.
* Seeds fix `torch`, `numpy` and `random`; bitwise GPU determinism is not claimed.
* The `encode` steps are one-off. Re-running them overwrites only files under
  `IAE_CACHE_ROOT`.

## License

MIT — see [LICENSE](LICENSE). 
