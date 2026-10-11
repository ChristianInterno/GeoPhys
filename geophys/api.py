"""
Python interface.

    import geophys

    geophys.rate(["a.mp4", "b.mp4", "c.mp4"])     # one score per video (DINOv3): 0.5 typical,
                                                   # higher = less plausible
    geophys.compare("a.mp4", "b.mp4")             # which of the two violates physics
    geophys.pick(["c0.mp4", "c1.mp4", "c2.mp4"])  # the most plausible of several candidates

compare and pick use the configurations in configs/ (any other with config=...). They
extract features once and cache them under GEOPHYS_STORES, keyed by the set of videos:
calling again on the same videos is fast, and many pairs or candidate sets should be
passed in one call rather than in a loop.
"""
import os
import json
import hashlib

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(HERE, "configs")


def _device(device):
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _config(config, default):
    path = config or os.path.join(CONFIGS, default)
    if not os.path.exists(path):
        raise FileNotFoundError("configuration %s not found; install GeoPhys from a clone of the "
                                "repository (pip install -e .) or pass config=" % path)
    return json.load(open(path))


def _values(specs, vids, meta, device):
    """Every member's value on every video, extracting what the members need."""
    from .extract import ensure_store
    from .members import compute_member, _CACHE
    tag = "api" + hashlib.md5("\n".join(vids).encode()).hexdigest()[:12]
    need = {}
    for m in specs:
        need.setdefault((m["backbone"], m["prep"]), set()).add(str(m["layer"]))
    for (backbone, prep), layers in sorted(need.items()):
        ensure_store(tag, backbone, prep, vids, meta, sorted(layers), device)
    _CACHE.clear()
    return [compute_member(s, tag) for s in specs]


def rate(videos, reference=None, backbone="dinov3", layer=None, prep="direct", device=None):
    """One plausibility score per video. Each video's trajectory through the encoder gives
    five signals (curvature, angle consistency, speed variation, acceleration, prediction
    residual); each becomes a percentile among `videos`, or among `reference` videos if
    given, and the score is their mean. Returns one dict per video: video, score, signals."""
    from .backbones import get_backbone, READOUT
    from .score import video_signals, percentiles, SIGNALS
    videos = [os.path.abspath(v) for v in videos]
    reference = [os.path.abspath(v) for v in reference] if reference else []
    if len(videos) < 2 and not reference:
        raise ValueError("scores are relative: give several videos, or reference videos")
    layer, device = layer or READOUT[backbone], _device(device)
    ext = get_backbone(backbone, device)
    sig = {v: video_signals(ext, v, layer, prep, device) for v in dict.fromkeys(videos + reference)}
    row = lambda v: [sig[v][k] if sig[v] else np.nan for k in SIGNALS]
    X = np.array([row(v) for v in videos])
    score = np.nanmean(percentiles(X, np.array([row(v) for v in reference]) if reference else X), 1)
    return [dict(video=v, score=float(s), **(sig[v] or {})) for v, s in zip(videos, score)]


def compare(a, b=None, config=None, dataset=None, device=None):
    """Which video of a pair violates physics. compare(a, b) for one pair, compare([(a, b), ...])
    for many. Returns {"implausible": path, "plausible": path, "confidence": 0-1} per pair."""
    from .ensemble import vote_pairs
    single = b is not None
    pairs = [(a, b)] if single else list(a)
    cfg = _config(config, "geophys_plus_detection.json")
    blocks = list(cfg["datasets"])
    specs = cfg["datasets"][dataset or (blocks[0] if len(blocks) == 1 else "likephys")]["members"]
    vids, where = [], {}
    for p in pairs:
        for v in map(os.path.abspath, p):
            where.setdefault(v, len(vids))
            if where[v] == len(vids):
                vids.append(v)
    vi = np.array([where[os.path.abspath(p[0])] for p in pairs])
    ii = np.array([where[os.path.abspath(p[1])] for p in pairs])
    meta = dict(pairs=[{"group": str(k), "vi": int(x), "ii": int(y)} for k, (x, y) in enumerate(zip(vi, ii))])
    vals = _values(specs, vids, meta, _device(device))
    votes = vote_pairs([np.asarray(v[ii] - v[vi], np.float64) for v in vals], specs)
    cells = max(len({(m["signal"], m["backbone"]) for m in specs}), 1)
    out = []
    for (x, y), vote in zip(pairs, votes):
        bad, good = (y, x) if vote > 0 else (x, y)
        out.append({"implausible": bad, "plausible": good, "confidence": float(min(abs(vote) / cells, 1.0))})
    return out[0] if single else out


def pick(candidates, config=None, device=None):
    """The most plausible of several candidates. pick([c0, c1, ...]) returns the index of the
    chosen candidate; pick([[c0, c1, ...], [d0, d1, ...]]) returns one index per set."""
    from .ensemble import fuse
    single = isinstance(candidates[0], str)
    sets = [candidates] if single else candidates
    cfg = _config(config, "geophys_physicsiq.json")
    vids, idx = [], []
    for s, cands in enumerate(sets):
        for c, v in enumerate(cands):
            idx.append({"scen": "set%d" % s, "cand": c, "vi": len(vids)})
            vids.append(os.path.abspath(v))
    meta = dict(scenarios=["set%d" % s for s in range(len(sets))], index=idx)
    K = max(len(c) for c in sets)
    members = []
    for v in _values(cfg["members"], vids, meta, _device(device)):
        M = np.full((len(sets), K), np.nan)
        for e in idx:
            M[int(e["scen"][3:]), e["cand"]] = v[e["vi"]]
        members.append(M)
    S = fuse(members, cfg["fusion"])
    S[np.all([np.isnan(M) for M in members], 0)] = np.inf     # sets with fewer candidates
    chosen = [int(i) for i in np.argmin(S, 1)]
    return chosen[0] if single else chosen
