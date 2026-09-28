"""Example 1: aggregate the per-run results into the paper table.

Reads the `RESULT,<encoder>,<norm>,<seed>,<all>,<notched>,<smooth>` lines printed by
train_ex1.py (run_ex1.sh collects them into results/ex1_runs.csv) and reports, for
each (encoder, normalization), the mean +- population standard deviation over seeds.

  python aggregate_ex1.py [results/ex1_runs.csv] [--latex]
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ENCODERS = ("mfe2d", "mfe1d", "iae_cosine", "iae_legendre")
NORMS = ("none", "perdim", "lowfreq")
LABEL = {"mfe2d": "2D-MFE", "mfe1d": "1D-MFE", "iae_cosine": "IAE (cosine)",
         "iae_legendre": "IAE (Legendre)"}
NLABEL = {"none": "none", "perdim": "per-dim", "lowfreq": "low-freq"}


def read(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"no results file {path}\nrun `bash run_ex1.sh` first, or pass a CSV "
                         "of RESULT lines as the first argument")
    runs = {}
    for line in path.read_text().splitlines():
        if not line.startswith("RESULT,"):
            continue
        _, enc, norm, seed, a, n, s = line.strip().split(",")
        runs.setdefault((enc, norm), {})[int(seed)] = (float(a), float(n), float(s))
    return {k: np.array([v[s] for s in sorted(v)]) for k, v in runs.items()}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else HERE / "results" / "ex1_runs.csv"
    runs = read(path)
    if not runs:
        raise SystemExit(f"no RESULT lines in {path}")
    latex = "--latex" in sys.argv
    print(f"{'encoder':15s} {'norm':9s} {'runs':>4s} {'overall':>17s} {'notched':>17s} {'smooth':>17s}")
    for enc in ENCODERS:
        for norm in NORMS:
            R = runs.get((enc, norm))
            if R is None:
                continue
            m, sd = R.mean(0), R.std(0)          # population std (ddof=0), as in the paper
            if latex:
                cells = " & ".join(f"${m[i]:.4f}{{\\pm}}{sd[i]:.4f}$" for i in range(3))
                print(f"{LABEL[enc]} & {NLABEL[norm]} & {cells} \\\\")
            else:
                cells = "  ".join(f"{m[i]:.4f} +- {sd[i]:.4f}" for i in range(3))
                print(f"{LABEL[enc]:15s} {NLABEL[norm]:9s} {len(R):>4d} {cells}")


if __name__ == "__main__":
    main()
