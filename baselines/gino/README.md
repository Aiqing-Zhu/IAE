# GINO baseline (Example 1)

The last row of Table 1 is GINO, trained on the **same data split** as the MIONet
rows with the **same five-seed protocol**.

## Implementation

The model is the reference implementation of GINO provided by the
[NeuralOperator](https://github.com/neuraloperator/neuraloperator) library,
version `2.0.0` at commit `93d3f06`, used unmodified. Install it with

```bash
pip install neuraloperator==2.0.0          # pulls in tensorly, tensorly-torch
```

or point the script at a checkout of the repository:

```bash
export NEURALOP_PATH=/path/to/neuraloperator
```

The library is a separate dependency from the rest of this repository — the IAE
code does not need it. If you use it, cite both the GINO paper and the library.

## Running

```bash
python gino_baseline.py build-cache               # one-off, ~4.8 GB of point clouds
python gino_baseline.py verify-cache              # bitwise check against the raw data
for s in 0 1 2 3 4; do
  python gino_baseline.py train $s 120000         # prefix with CUDA_VISIBLE_DEVICES=<gpu>
done                                              # to place concurrent seeds on given GPUs
python gino_baseline.py summarize 120000          # -> the Table-1 GINO row
```

**Resources.** The cache lands in `$IAE_CACHE_ROOT/gino/gino_inputs.pkl` (default
`cache/gino/`, ~4.8 GB — it is *not* included in the ~850 MB the main README quotes
for the IAE caches). Training moves all of it onto the GPU, so a seed needs roughly
14 GB of device memory, and it re-reads the 2.2 GB of raw `.npz` on every launch.
One update takes about 0.56 s on an L40S, so 120000 updates is ~19 h per seed and
the five seeds together took 93.9 GPU-hours.

Weights, metrics, the data checksums and the full convergence history are written
to `results/gino-s<seed>-it<iters>.{pt,json}`.

`paper_runs/` holds the five result files of the runs reported in the paper, so the
Table-1 row can be checked without spending the 94 GPU-hours again:

```bash
python gino_baseline.py summarize 120000 paper_runs
```

## What is and is not matched

| | matched to the MIONet rows |
|---|---|
| data split, sample order | yes — the same `raw_train.npz → raw_train_b.npz → raw_test.npz`, no shuffling, so sample `i` is the same domain |
| evaluation | yes — relative L2 at the same mesh nodes, same notched / smooth split |
| protocol | yes — five seeded runs, mean ± population std |
| architecture, hyperparameter tuning | **no** |
| training budget | **no** — see below |

Each GINO update consumes one geometry, because the reference implementation takes
a single input geometry per forward pass. The reported runs use `1.2e5` sample
presentations against `7.7e6` for the MIONet rows. They are nevertheless trained to
convergence: for every seed the overall test error changes by less than `3.1e-4`
between updates 115000 and 120000 — the statistic the paper's appendix quotes, which
`summarize` prints so it can be checked against the retained histories (it also
prints the wider three-evaluation window, where the largest move is `6.8e-4`).
Matching the number of presentations exactly is not feasible — at the measured
0.56 s per update, `7.7e6` presentations is about 50 days per seed.
