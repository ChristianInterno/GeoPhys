"""
Which of a pair of videos violates physics.

Point this at a directory of paired videos and it says which one of each pair looks
physically implausible, using a frozen configuration. Nothing is trained: each member
scores both videos, the signed difference votes, and the majority decides.

    my_pairs/
      case_a/
        a.mp4
        b.mp4
      case_b/
        ...

    python -m geophys.pairs --videos my_pairs/ --out verdicts.csv

Each sub-directory must hold exactly two videos. They are taken in sorted filename
order, so name them however you like; the output names the file judged implausible.

If you already know which is which and want to measure accuracy, name them so that
the plausible one sorts first (for example `a_valid.mp4` / `b_invalid.mp4`) and pass
--labelled: the command then reports how often it picked the second file.

The configuration defaults to GeoPhys+ (configs/geophys_plus_detection.json); --config
takes any other, including one made by geophys.calibrate. Extraction runs once per
(backbone, preprocessing) the configuration needs and is cached under GEOPHYS_STORES,
so a second run over the same videos is fast.
"""
import os
import csv
import json
import argparse

import numpy as np

from .extract import STORES, ensure_store, manifest_from_config
from .members import compute_member, _CACHE
from .ensemble import vote_pairs, macro_micro

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(HERE, "configs")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".webm", ".mkv")
DEFAULT_CONFIG = os.path.join(CONFIGS, "geophys_plus_detection.json")


def index_pairs(root):
    """<root>/<case>/{first,second}.mp4 -> (cases, video paths, index entries)."""
    cases = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not cases:
        raise SystemExit("no case sub-directories found in %s" % root)
    vids, pairs, skipped = [], [], []
    for c in cases:
        files = sorted(f for f in os.listdir(os.path.join(root, c))
                       if f.lower().endswith(VIDEO_EXT))
        if len(files) != 2:
            skipped.append((c, len(files)))
            continue
        first, second = files
        pairs.append({"case": c, "first": first, "second": second,
                      "vi": len(vids), "ii": len(vids) + 1})
        vids += [os.path.abspath(os.path.join(root, c, f)) for f in (first, second)]
    if skipped:
        print("skipped %d directories without exactly two videos (e.g. %s: %d files)"
              % (len(skipped), skipped[0][0], skipped[0][1]))
    print("%d pairs, %d videos" % (len(pairs), len(vids)))
    if not pairs:
        raise SystemExit("nothing to do")
    return pairs, vids


def index_pair_list(pairs_file, data_root=""):
    """Benchmark pair list (data/<dataset>_pairs.json) -> (pairs, video paths).
    The valid video of each pair becomes the first file, the invalid one the second."""
    items = json.load(open(pairs_file))
    vids, where, pairs = [], {}, []
    def slot(path):
        if data_root and not os.path.isabs(path):
            path = os.path.join(data_root, path)
        path = os.path.abspath(path)          # the same video, however --data-root is written
        if path not in where:
            where[path] = len(vids)
            vids.append(path)
        return where[path]
    for n, e in enumerate(items):
        pairs.append({"case": e["invalid"], "first": e["valid"], "second": e["invalid"],
                      "group": e.get("group", n), "vi": slot(e["valid"]), "ii": slot(e["invalid"])})
    missing = [v for v in vids if not os.path.exists(v)]
    if missing:
        raise SystemExit("%d of %d videos not found, e.g. %s\n"
                         "is --data-root the folder the paths in %s start from?"
                         % (len(missing), len(vids), missing[0], pairs_file))
    print("%d pairs, %d distinct videos" % (len(pairs), len(vids)))
    return pairs, vids


def ensure_stores(cfg, dataset, vids, pairs, device, tag):
    """Extract whatever (backbone, preprocessing) combinations the configuration needs."""
    need = {}
    for m in cfg["datasets"][dataset]["members"]:
        need.setdefault((m["backbone"], m["prep"]), set()).add(str(m["layer"]))
    meta_pairs = [{"group": p.get("group", p["case"]), "vi": p["vi"], "ii": p["ii"]}
                  for p in pairs]
    for (backbone, prep), layers in sorted(need.items()):
        ensure_store(tag, backbone, prep, vids, dict(pairs=meta_pairs), sorted(layers), device)


def main():
    ap = argparse.ArgumentParser(description="GeoPhys — which of a pair violates physics")
    ap.add_argument("--videos", help="directory of <case>/<two videos>")
    ap.add_argument("--pairs", help="benchmark pair list, e.g. data/likephys_pairs.json; "
                    "reports macro and micro accuracy")
    ap.add_argument("--data-root", default="",
                    help="directory that relative paths in --pairs start from")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--dataset", default=None,
                    help="which member list of the configuration to use "
                         "(the shipped configurations have likephys and intphys2)")
    ap.add_argument("--out", default="verdicts.csv")
    ap.add_argument("--labelled", action="store_true",
                    help="assume the second file of each pair is the implausible one "
                         "and report accuracy")
    ap.add_argument("--tag", default="pairs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    if not (a.videos or a.pairs):
        ap.error("give --videos (your own pairs) or --pairs (a benchmark pair list)")
    cfg = json.load(open(a.config))
    # which member list: the only one in the configuration, else from the pair-list
    # name, else LikePhys
    blocks = list(cfg["datasets"])
    dataset = a.dataset or (blocks[0] if len(blocks) == 1 else
                            "intphys2" if a.pairs and "intphys2" in os.path.basename(a.pairs)
                            else "likephys")
    block = cfg["datasets"][dataset]
    print("configuration: %s / %s (%d members)" % (cfg.get("method", "?"), dataset, len(block["members"])))
    pairs, vids = (index_pair_list(a.pairs, a.data_root) if a.pairs
                   else index_pairs(a.videos))

    if a.plan:
        for (store, prep), layers in sorted(manifest_from_config(a.config).items()):
            print("  %-16s %-7s layers: %s" % (store, prep, ",".join(layers)))
        return

    print("stores (%s):" % STORES)
    ensure_stores(cfg, dataset, vids, pairs, a.device, a.tag)
    _CACHE.clear()

    vi = np.array([p["vi"] for p in pairs])
    ii = np.array([p["ii"] for p in pairs])
    specs = block["members"]
    deltas = []
    for k, spec in enumerate(specs, 1):
        print("  [%3d/%3d] %-8s %-10s %-5s" % (k, len(specs), spec["backbone"],
                                               spec["signal"], spec["agg"]),
              end="\r", flush=True)
        v = compute_member(spec, a.tag)
        # positive delta = the second file scores as less plausible
        deltas.append(np.asarray(v[ii] - v[vi], np.float64))
    print("  computed %d members%s" % (len(specs), " " * 30))

    # the same vote as detect.py
    group = np.array([str(p.get("group", p["case"])) for p in pairs])
    votes = vote_pairs(deltas, specs)
    second_is_violated = votes > 0
    # confidence: the margin of the cells' vote, as a fraction of the cells
    conf = np.clip(np.abs(votes) / max(len({(m["signal"], m["backbone"]) for m in specs}), 1), 0, 1)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "judged_implausible", "judged_plausible", "confidence"])
        for i, p in enumerate(pairs):
            bad, good = ((p["second"], p["first"]) if second_is_violated[i]
                         else (p["first"], p["second"]))
            w.writerow([p["case"], bad, good, "%.3f" % conf[i]])
    print("wrote %s (%d pairs)" % (a.out, len(pairs)))

    if a.pairs:
        macro, micro = macro_micro(second_is_violated, group)
        print("accuracy on %s: %.2f%% macro (mean over scenarios), %.2f%% micro "
              "(over pairs)" % (os.path.basename(a.pairs), macro, micro))
    elif a.labelled:
        acc = 100.0 * float(second_is_violated.mean())
        print("accuracy (assuming the second file is implausible): %.2f%%" % acc)


if __name__ == "__main__":
    main()
