"""
One plausibility score per video, from a single frozen encoder (DINOv3 by default).

For benchmarks that rate every generated video on its own rather than in pairs or
among candidates, for example the physical-realism axis of a 3D/4D world-generation
benchmark such as 4DWorldBench. Give one folder per model (or per setting):

    python -m geophys.score --videos outputs/model_a outputs/model_b --out scores.csv

Each video is decoded to at most 60 frames and encoded frame by frame with DINOv3
ViT-L/16; the patch tokens of block 18 are averaged per frame, giving a trajectory of
frame features. Five geometric signals are read off that trajectory: curvature, angle
consistency, speed variation, acceleration and the linear-prediction residual. Each
signal is turned into a percentile among all the videos scored together, or among
--reference videos (for example real footage), and the GeoPhys score of a video is the
mean of its five percentiles: 0.5 is typical, higher means a less regular trajectory,
which is what physically implausible motion produces.

Scores are relative to the set they are standardised over, so compare models scored in
one run (or against the same --reference). Videos with very different amounts of
motion or very different content are less comparable than outputs for the same prompts.

Output: <out> with one row per video (folder, file, score and each signal), and a
summary per folder (mean score, standard error, number of videos), lowest first.
Per-video signals are cached in <out>.signals.jsonl, so an interrupted run resumes.
"""
import os
import csv
import json
import argparse

import numpy as np

from .backbones import get_backbone, load_frames, prep_frames, READOUT
from .extract import _forward
from .signals import gram_signals

VIDEO_EXT = (".mp4", ".avi", ".mov", ".webm", ".mkv")
SIGNALS = ["curv", "angcons", "speedcv", "accel", "prederr"]


def list_videos(folder):
    out = []
    for root, _, files in os.walk(folder):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(VIDEO_EXT)]
    return sorted(out)


def video_signals(ext, path, layer, prep, device):
    """The geometric signals of one video's mean-token trajectory at one layer."""
    frames = load_frames(path)
    if len(frames) < 4:
        return None
    pooled, _ = _forward(ext, prep_frames(frames, prep), device)
    z = pooled[layer]["mean"].double()
    g = (z @ z.T).cpu().numpy()[None]
    s = gram_signals(g, ars=[])
    return {k: float(s[k][0]) for k in SIGNALS}


def percentiles(X, R):
    """Each column of X as a percentile (0-1) among the same column of R."""
    out = np.full(X.shape, np.nan)
    for k in range(X.shape[1]):
        r = np.sort(R[:, k][~np.isnan(R[:, k])])
        if len(r):
            p = (np.searchsorted(r, X[:, k], "left") + np.searchsorted(r, X[:, k], "right")) / (2.0 * len(r))
            out[:, k] = np.where(np.isnan(X[:, k]), np.nan, p)
    return out


def main():
    ap = argparse.ArgumentParser(description="GeoPhys - one plausibility score per video")
    ap.add_argument("--videos", nargs="+", required=True, help="one or more folders of videos")
    ap.add_argument("--reference", default=None,
                    help="folder of videos to standardise against (default: all scored videos)")
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--backbone", default="dinov3", choices=sorted(READOUT))
    ap.add_argument("--layer", default=None, help="readout layer (default: 18 for DINOv3)")
    ap.add_argument("--prep", default="direct", choices=["direct", "hf"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    layer = a.layer or READOUT[a.backbone]

    folders = [os.path.abspath(f) for f in a.videos]
    jobs = [(f, v) for f in folders for v in list_videos(f)]
    ref = list_videos(os.path.abspath(a.reference)) if a.reference else []
    if not jobs:
        raise SystemExit("no videos found in %s" % ", ".join(a.videos))
    print("%d videos in %d folders%s; %s block %s, %s preprocessing"
          % (len(jobs), len(folders), ", %d reference videos" % len(ref) if ref else "",
             a.backbone, layer, a.prep), flush=True)

    cache_path = a.out + ".signals.jsonl"
    setting = "%s|%s|%s" % (a.backbone, layer, a.prep)
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            if r.get("setting") == setting:
                cache[r["path"]] = r["signals"]
    todo = [v for v in [v for _, v in jobs] + ref if v not in cache]
    if todo:
        from tqdm import tqdm
        ext = get_backbone(a.backbone, a.device)
        with open(cache_path, "a") as fh:
            for v in tqdm(todo):
                s = video_signals(ext, v, layer, a.prep, a.device)
                cache[v] = s
                fh.write(json.dumps({"path": v, "setting": setting, "signals": s}) + "\n")
                fh.flush()

    X = np.array([[cache[v][k] if cache[v] else np.nan for k in SIGNALS] for _, v in jobs])
    R = np.array([[cache[v][k] if cache[v] else np.nan for k in SIGNALS] for v in ref]) if ref else X
    score = np.nanmean(percentiles(X, R), 1)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "video", "geophys_score"] + SIGNALS)
        for (f, v), s, x in zip(jobs, score, X):
            w.writerow([os.path.basename(f), os.path.relpath(v, f), "%.4f" % s] + ["%.6g" % t for t in x])
    print("wrote %s" % a.out)

    rows = []
    for f in folders:
        s = np.array([sc for (g, _), sc in zip(jobs, score) if g == f])
        s = s[~np.isnan(s)]
        if len(s):
            rows.append((float(np.mean(s)), float(np.std(s, ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
                         len(s), os.path.basename(f)))
    print("\n%-32s %10s %8s %6s" % ("folder (lower = more plausible)", "score", "s.e.", "n"))
    for m, se, n, f in sorted(rows):
        print("%-32s %10.3f %8.3f %6d" % (f, m, se, n))
    skipped = sum(1 for _, v in jobs if not cache[v])
    if skipped:
        print("\n%d videos had fewer than 4 frames and were skipped" % skipped)


if __name__ == "__main__":
    main()
