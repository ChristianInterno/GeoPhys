"""
Physical-violation detection.

Each dataset is a set of paired videos: one physically plausible, one implausible.
A configuration names a set of members; each member scores both videos of a pair, and
the vote decides which of the two is the implausible one. Each pair is decided from its
two videos alone.

    python -m geophys.detect --members members/   # the paper's tables, from the archive
    python -m geophys.detect --stores STORES       # the same, from feature stores
    python -m geophys.detect --method geophys --dataset likephys
"""
import os
import sys
import json
import argparse

import numpy as np

from .ensemble import vote_pairs, macro_micro
from .members import detection_members

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(HERE, "configs")
METHODS = {"geophys": "GeoPhys", "geophys_plus": "GeoPhys+"}
DATASETS = ["likephys", "intphys2"]


def run(method, dataset, root=None, cache=None):
    """Macro and micro accuracy of one configuration on one dataset."""
    cfg = json.load(open(os.path.join(CONFIGS, "%s_detection.json" % method)))
    specs = cfg["datasets"][dataset]["members"]
    npz = os.path.join(cache, "%s_%s.npz" % (method, dataset)) if cache else None
    if npz and os.path.exists(npz):
        z = np.load(npz, allow_pickle=False)
        deltas, group = z["deltas"], z["group"]
    else:
        deltas, group = detection_members(cfg, dataset, root, cache=cache)
    return macro_micro(vote_pairs(deltas, specs) > 0, group)


def main():
    ap = argparse.ArgumentParser(description="GeoPhys - physical-violation detection")
    ap.add_argument("--method", choices=list(METHODS), action="append")
    ap.add_argument("--dataset", choices=DATASETS, action="append")
    ap.add_argument("--members", default=None,
                    help="directory of precomputed member arrays (created if absent)")
    ap.add_argument("--stores", default=None, help="root directory of feature stores")
    a = ap.parse_args()

    datasets = a.dataset or DATASETS
    rows = []
    for m in a.method or list(METHODS):
        cells = {}
        for ds in datasets:
            try:
                if not (a.members and os.path.exists(os.path.join(a.members, "%s_%s.npz" % (m, ds)))):
                    print("  computing %s on %s from feature stores..." % (METHODS[m], ds),
                          file=sys.stderr, flush=True)
                cells[ds] = "%.2f / %.2f" % run(m, ds, a.stores, a.members)
            except FileNotFoundError as e:
                cells[ds] = "missing store"
                print("    %s" % e, file=sys.stderr)
        rows.append((METHODS[m], cells))

    w = 26
    print()
    print("Physical-violation detection (macro / micro %)")
    print("-" * (w + 20 * len(datasets)))
    print("%-*s" % (w, "Method") + "".join("%-20s" % d for d in datasets))
    print("-" * (w + 20 * len(datasets)))
    for name, cells in rows:
        print("%-*s" % (w, name) + "".join("%-20s" % cells.get(d, "-") for d in datasets))
    print()


if __name__ == "__main__":
    main()
