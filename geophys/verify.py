"""
Best-of-N verification on Physics-IQ.

For each of the 198 scenarios a generator produced 16 candidate continuations. The
verifier scores all sixteen from frozen-backbone trajectory geometry and picks one per
scenario; the number reported is the mean Physics-IQ score of the picks, read from the
per-candidate score table in data/.

    python -m geophys.verify --members members/            # all three methods
    python -m geophys.verify --method geophys_plus          # one method
    python -m geophys.verify --members members/ --export picks/   # also write the picks
"""
import os
import sys
import json
import argparse

import numpy as np

from .ensemble import fuse
from .members import verification_members

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(HERE, "configs")
GT = os.path.join(HERE, "data", "magi_v2v_16_per_seed_24fps.csv")
METHODS = ["geophys", "geophys_plus", "geophys_plusplus"]


def load_gt(path=GT):
    import pandas as pd
    G = pd.read_csv(path)
    return G, G.iloc[:, 1:17].values.astype(np.float64)


def score_selection(T, sel):
    ar = np.arange(T.shape[0])
    v = T[ar, sel]
    return float(np.nanmean(v[~np.isnan(v)]))


def run(method, gt_path=GT, root=None, cache=None):
    cfg = json.load(open(os.path.join(CONFIGS, "%s_physicsiq.json" % method)))
    _, T = load_gt(gt_path)

    npz = os.path.join(cache, "%s_physicsiq.npz" % method) if cache else None
    if npz and os.path.exists(npz):               # released member arrays
        members = list(np.load(npz, allow_pickle=False)["members"].astype(np.float64))
    else:
        members = verification_members(cfg, gt_path, root, cache=cache)
    sel = np.argmin(fuse(members, cfg["fusion"]), 1)
    return score_selection(T, sel), sel, len(members)


def export_selection(sel, gt_path, out_dir, candidates):
    """Symlink the selected candidate of each scenario into out_dir."""
    import pandas as pd
    G = pd.read_csv(gt_path)
    os.makedirs(out_dir, exist_ok=True)
    rows, made, missing = [], 0, []
    for i, sid in enumerate(G["scenario_id"]):
        ci = int(sel[i])
        src = os.path.join(candidates, sid, "candidate_%03d.mp4" % ci)
        dst = os.path.join(out_dir, sid + ".mp4")
        if os.path.exists(dst):
            made += 1
        elif os.path.exists(src):
            os.symlink(os.path.abspath(src), dst); made += 1
        else:
            missing.append("%s (candidate %d)" % (sid, ci))
        rows.append({"scenario_id": sid, "cand_idx": ci})
    pd.DataFrame(rows).to_csv(out_dir + "_manifest.csv", index=False)
    if missing:
        print("    WARNING: %d selected videos not found under %s, e.g. %s"
              % (len(missing), candidates, ", ".join(missing[:3])), file=sys.stderr)
    return made


def main():
    ap = argparse.ArgumentParser(description="GeoPhys — best-of-16 verification")
    ap.add_argument("--method", choices=METHODS, action="append")
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--members", default=None,
                    help="directory of precomputed member arrays (created if absent)")
    ap.add_argument("--stores", default=None)
    ap.add_argument("--export", default=None,
                    help="directory to write the selected videos into")
    ap.add_argument("--candidates", default=os.path.join(HERE, "videos", "v2v_candidates"),
                    help="with --export: directory of <scenario>/candidate_XXX.mp4")
    a = ap.parse_args()

    _, T = load_gt(a.gt)
    baseline = float(np.nanmean(T[:, 0]))
    oracle = float(np.nanmean(np.nanmax(T, 1)))
    label = {"geophys": "GeoPhys",
             "geophys_plus": "GeoPhys+",
             "geophys_plusplus": "GeoPhys++"}

    print()
    print("Best-of-16 verification, Physics-IQ (MAGI-1 24B V2V, %d scenarios)" % T.shape[0])
    print("-" * 41)
    print("%-28s %12s" % ("Selector", "Physics-IQ"))
    print("-" * 41)
    print("%-28s %12.2f" % ("Baseline (first candidate)", baseline))
    for m in a.method or METHODS:
        try:
            if not (a.members and os.path.exists(
                    os.path.join(a.members, "%s_physicsiq.npz" % m))):
                print("  computing %s from feature stores..." % label.get(m, m),
                      file=sys.stderr, flush=True)
            score, sel, n = run(m, a.gt, a.stores, a.members)
            print("%-28s %12.2f" % (label.get(m, m), score))
            if a.export:
                out = os.path.join(a.export, m)
                made = export_selection(sel, a.gt, out, a.candidates)
                print("%-28s   -> %d videos in %s" % ("", made, out))
        except FileNotFoundError as e:
            print("%-28s %12s   (%s)" % (label.get(m, m), "missing", e))
    print("%-28s %12.2f" % ("Oracle (best candidate)", oracle))
    print()
    print("To score the picks with the official Physics-IQ code (original and Verified")
    print("metrics), write them out with --export DIR and run the benchmark's")
    print("run_physics_iq.py with --input_folders DIR/<method>.")
    print()


if __name__ == "__main__":
    main()
