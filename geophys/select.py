"""
Best-of-N verification on your own videos.

Point this at a directory of candidate continuations and it tells you which one is
most physically plausible, using a frozen configuration. No ground truth needed and
nothing is trained: the score is a geometric property of each video's trajectory
through frozen image encoders, normalised within each scenario.

    my_videos/
      scenario_a/
        candidate_000.mp4
        candidate_001.mp4
        ...
      scenario_b/
        ...

    python -m geophys.select --videos my_videos/ --out picks.csv

The configuration defaults to GeoPhys (configs/geophys_physicsiq.json); --config takes
any other. Extraction runs once per (backbone, preprocessing) the configuration needs
and is cached under GEOPHYS_STORES, so a second run over the same videos is fast.

Scores are comparable *within* a scenario, not across scenarios or against an
absolute threshold: GeoPhys ranks candidates, it does not certify a single video.
"""
import os
import csv
import json
import argparse

import numpy as np

from .ensemble import fuse
from .extract import STORES, ensure_store, manifest_from_config
from .members import open_store, _CACHE, compute_member

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(HERE, "configs")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".webm", ".mkv")


def index_videos(root):
    """<root>/<scenario>/<candidate>.<ext> -> (scenarios, video paths, index entries)."""
    scen = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not scen:
        raise SystemExit("no scenario sub-directories found in %s" % root)
    vids, idx = [], []
    for s in scen:
        files = sorted(f for f in os.listdir(os.path.join(root, s))
                       if f.lower().endswith(VIDEO_EXT))
        for c, f in enumerate(files):
            idx.append({"scen": s, "cand": c, "vi": len(vids)})
            vids.append(os.path.join(root, s, f))
    counts = {}
    for e in idx:
        counts[e["scen"]] = counts.get(e["scen"], 0) + 1
    n = max(counts.values())
    odd = {k: v for k, v in counts.items() if v != n}
    print("%d scenarios, %d videos (%d candidates each%s)"
          % (len(scen), len(vids), n,
             "; %d scenarios differ" % len(odd) if odd else ""))
    return scen, vids, idx, n


def ensure_stores(cfg, vids, scen, idx, device, tag):
    """Extract whatever (backbone, preprocessing) combinations the configuration needs."""
    need = {}
    for m in cfg["members"]:
        need.setdefault((m["backbone"], m["prep"]), set()).add(str(m["layer"]))
    for (backbone, prep), layers in sorted(need.items()):
        ensure_store(tag, backbone, prep, vids, dict(scenarios=scen, index=idx),
                     sorted(layers), device)


def main():
    ap = argparse.ArgumentParser(description="GeoPhys — pick the most plausible candidate")
    ap.add_argument("--videos", required=True,
                    help="directory of <scenario>/<candidate>.mp4")
    ap.add_argument("--config", default=os.path.join(CONFIGS, "geophys_physicsiq.json"))
    ap.add_argument("--out", default="picks.csv")
    ap.add_argument("--gt", default=None,
                    help="optional Physics-IQ score table, e.g. "
                         "data/magi_v2v_16_per_seed_24fps.csv: also report the score of the picks")
    ap.add_argument("--export", default=None,
                    help="directory to symlink the chosen videos into")
    ap.add_argument("--tag", default="user",
                    help="prefix for the extracted stores (lets you keep several sets apart)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plan", action="store_true",
                    help="print what would be extracted and exit")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    print("configuration: %s (%d members, fusion %s)"
          % (cfg.get("method", "?"), len(cfg["members"]), cfg["fusion"]))
    scen, vids, idx, NC = index_videos(a.videos)

    if a.plan:
        for (store, prep), layers in sorted(manifest_from_config(a.config).items()):
            print("  %-16s %-7s layers: %s" % (store, prep, ",".join(layers)))
        return

    print("stores (%s):" % STORES)
    ensure_stores(cfg, vids, scen, idx, a.device, a.tag)
    _CACHE.clear()

    sidx = {s: i for i, s in enumerate(scen)}
    members = []
    specs = cfg["members"]
    for k, spec in enumerate(specs, 1):
        print("  [%3d/%3d] %-8s %-10s %-6s" % (k, len(specs), spec["backbone"],
                                               spec["signal"], spec["agg"]),
              end="\r", flush=True)
        # stores are named <tag>_<store>_<prep>
        st = open_store(a.tag, spec["backbone"], spec["prep"])
        v = compute_member(spec, a.tag)
        M = np.full((len(scen), NC), np.nan)
        for e in st.meta["index"]:
            c = int(e["cand"])
            if e["scen"] in sidx and 0 <= c < NC:
                M[sidx[e["scen"]], c] = v[e["vi"]]
        members.append(M)
    print("  computed %d members%s" % (len(specs), " " * 30))

    S = fuse(members, cfg["fusion"])
    sel = np.argmin(S, 1)
    margin = np.sort(S, 1)
    gap = margin[:, 1] - margin[:, 0] if S.shape[1] > 1 else np.zeros(len(sel))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "chosen_candidate", "fused_score", "margin_to_runner_up"])
        for i, s in enumerate(scen):
            w.writerow([s, int(sel[i]), "%.4f" % S[i, sel[i]], "%.4f" % gap[i]])
    print("wrote %s (%d scenarios)" % (a.out, len(scen)))

    if a.gt:
        from .verify import load_gt, score_selection
        G, T = load_gt(a.gt)
        row = {sid: i for i, sid in enumerate(G["scenario_id"])}
        keep = np.array([i for i, sid in enumerate(scen) if sid in row], int)
        if len(keep) == 0:
            print("none of these scenarios is in %s" % a.gt)
        else:
            Ts = T[[row[scen[i]] for i in keep]]
            print("Physics-IQ on %d scenarios: picks %.2f | first candidate %.2f | "
                  "best candidate %.2f" % (len(keep), score_selection(Ts, sel[keep]),
                  float(np.nanmean(Ts[:, 0])), float(np.nanmean(np.nanmax(Ts, 1)))))

    if a.export:
        os.makedirs(a.export, exist_ok=True)
        made = 0
        for i, s in enumerate(scen):
            files = sorted(f for f in os.listdir(os.path.join(a.videos, s))
                           if f.lower().endswith(VIDEO_EXT))
            if int(sel[i]) >= len(files):
                continue
            src = os.path.abspath(os.path.join(a.videos, s, files[int(sel[i])]))
            dst = os.path.join(a.export, s + os.path.splitext(src)[1])
            if not os.path.exists(dst):
                os.symlink(src, dst)
                made += 1
        print("linked %d chosen videos -> %s" % (made, a.export))


if __name__ == "__main__":
    main()
