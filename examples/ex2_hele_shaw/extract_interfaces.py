"""Example 2: pull the interfaces out of the raw trajectories into a compact file.

The raw `traj_{train,test}.pkl` carry a full finite-element mesh and the pressure field at
every node -- about 1.2 MB per frame, of which the interface itself is under 0.2%.  The
operator only ever needs the interface (plus codes that are cached separately), so this
extracts

    loops   the interface polylines, verbatim from the raw file
    ncomp   the number of connected components (used to classify merges)
    time    the frame time

for both splits into `data/ex2_hele_shaw/interfaces.pkl` -- 19 MB instead of 4.5 GB.

That file is what the release distributes.  Together with the cached codes it is enough to
re-encode the geometry at a different order or grid (`train_evolution.py reencode`), because
`z_geom` is a function of the interface alone; the source code `z_f` and the auxiliary
pressure samples are carried over from the existing cache, as they do not depend on the
geometry discretization.

  python extract_interfaces.py
"""

from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iae.paths import EX2_DATA, need, out_dir                           # noqa: E402


def main():
    # This is an authoring tool: it is the one place that writes into the data
    # directory, turning the raw meshed trajectories into the file that ships.
    out = out_dir(EX2_DATA)
    bundle = {}
    for split in ("train", "test"):
        trajs = pickle.load(open(need(EX2_DATA / f"traj_{split}.pkl"), "rb"))
        bundle[split] = [[dict(loops=[np.asarray(l) for l in f["loops"]],
                               ncomp=int(f["ncomp"]), time=float(f["time"]))
                          for f in t] for t in trajs]
        n_frames = sum(len(t) for t in bundle[split])
        print(f"  {split}: {len(trajs)} trajectories, {n_frames} frames", flush=True)
        del trajs
    path = out / "interfaces.pkl"
    pickle.dump(bundle, open(path, "wb"), protocol=4)
    print(f"wrote {path}  ({path.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
