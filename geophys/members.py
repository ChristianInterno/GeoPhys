"""
Members: turning a configuration entry into numbers.

A member is one line of a configuration file, e.g.

    {"backbone": "cornet", "signal": "curv", "layer": "IT", "agg": "tok",
     "norm": "-", "tvar": "spd", "post": "top25", "window": "first", "prep": "hf"}

which reads as: take CORnet-S at its IT readout from the store extracted with hf
preprocessing, follow each spatial token's trajectory over the first half of the clip
using speed-normalised steps, compute curvature per token, and summarise the tokens
by the mean of the top 25.

The fields:
  agg     mean | cls | flat     one pooled trajectory per video (Gram of that feature)
          tok                   one trajectory per spatial token, summarised by post
  norm    l2                    cosine-normalised pooled trajectory; "-" for raw
  tvar    raw | l2 | spd        token steps: raw, of l2-normalised tokens, or speed-normalised
  post    how the per-token values are summarised (see signals.token_reduce)
  window  full | mid | first | last, or chg (self-change, see _compute_member)
  prep    direct | hf           frame preprocessing (see backbones.prep_frames)

"""
import os
import json
import hashlib

import numpy as np

from .signals import gram_signals, token_signals, token_reduce, EPS
from .extract import STORES

# Windows are fractions of the trajectory, resolved against the store's frame axis.
# On the 60-frame stores these give exactly slice(0,60), (15,45), (0,30), (30,60).
WINDOW_FRACTIONS = {"full": (0.0, 1.0), "mid": (0.25, 0.75),
                    "first": (0.0, 0.5), "last": (0.5, 1.0)}


def window_slice(name, n_frames):
    """full / mid / first / last: 0-100%, 25-75%, 0-50% and 50-100% of the trajectory."""
    if name not in WINDOW_FRACTIONS:
        raise KeyError("unknown window: %s" % name)
    lo, hi = WINDOW_FRACTIONS[name]
    return slice(int(round(lo * n_frames)), int(round(hi * n_frames)))


class Store:
    """One extracted store: <prefix>_<backbone>_<prep>/."""

    def __init__(self, path):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.layers = self.meta["layers"]
        self.atoms = self.meta.get("atoms", [])
        self._gram = None
        self._tok = {}

    @property
    def gram(self):
        if self._gram is None:
            self._gram = np.load(os.path.join(self.path, "gram.npy"), mmap_mode="r")
        return self._gram

    def tokens(self, tvar):
        """(sq, dot) memmaps for a token variant, or None if absent."""
        if tvar in self._tok:
            return self._tok[tvar]
        cands = [("t2_%s_sq.npy" % tvar, "t2_%s_dot.npy" % tvar)]
        if tvar == "raw":
            cands.append(("tok_sq.npy", "tok_dot.npy"))      # stores of earlier versions
        self._tok[tvar] = None
        for sq_name, dt_name in cands:
            sq_p, dt_p = os.path.join(self.path, sq_name), os.path.join(self.path, dt_name)
            if os.path.exists(sq_p) and os.path.exists(dt_p):
                self._tok[tvar] = (np.load(sq_p, mmap_mode="r"), np.load(dt_p, mmap_mode="r"))
                break
        return self._tok[tvar]


_CACHE = {}


def open_store(prefix, backbone, prep, root=None):
    key = (prefix, backbone, prep, root)
    if key not in _CACHE:
        path = os.path.join(root or STORES, "%s_%s_%s" % (prefix, backbone, prep))
        if not os.path.isdir(path):
            raise FileNotFoundError("missing store: %s" % path)
        _CACHE[key] = Store(path)
    return _CACHE[key]


def _pooled_gram(store, layer, atom, norm, window):
    """[N, T, T] Gram slice for one layer/atom, optionally cosine-normalised."""
    if atom not in store.atoms:
        raise KeyError("atom %s not in store %s" % (atom, store.path))
    G = np.array(store.gram[:, store.layers.index(layer), store.atoms.index(atom)], np.float64)
    G[G == 0] = np.nan
    if norm == "l2":
        d = np.sqrt(np.clip(np.diagonal(G, axis1=1, axis2=2), EPS, None))
        G = G / (d[:, :, None] * d[:, None, :])
    sl = window_slice(window, G.shape[1])
    return np.ascontiguousarray(G[:, sl, sl])


def _token_steps(store, layer, tvar, window):
    """Per-token squared steps [N, P, T'] and consecutive step products [N, P, T'-1]
    over a window; padding (zeros) becomes NaN."""
    tok = store.tokens(tvar if tvar != "-" else "raw")
    if tok is None:
        raise FileNotFoundError("no token arrays for variant %s in %s" % (tvar, store.path))
    SQ, DT = tok
    li = store.layers.index(layer)
    nf = SQ.shape[-1] + 1
    fi = np.arange(nf)[window_slice(window, nf)]
    # read the layer plane contiguously, then subset frames: strided reads across the
    # layer axis pull the whole array through the page cache
    sq = np.array(SQ[:, li], np.float32)[:, :, fi[0]:fi[-1]].copy()
    dt = np.array(DT[:, li], np.float32)[:, :, fi[0]:max(fi[-1] - 1, fi[0] + 1)].copy()
    sq[sq == 0] = np.nan
    dt[np.isnan(sq[..., :-1])] = np.nan
    return sq, dt


def member_key(spec, prefix):
    """Stable filename for a member: store prefix + the fields that define it."""
    fields = [prefix, spec["backbone"], spec["signal"], str(spec["layer"]), spec["agg"],
              str(spec.get("norm", "-")), str(spec.get("tvar", "-")), str(spec.get("post", "-")),
              spec.get("window", "full"), spec["prep"]]
    name = "__".join(fields)
    if len(name) > 120:
        name = name[:100] + "_" + hashlib.md5(name.encode()).hexdigest()[:8]
    return name + ".npy"


def compute_member(spec, prefix, root=None, cache=None):
    """One member, one value per video, in float32. If cache is a directory, the member
    is loaded from it when present and saved to it after computing."""
    if cache:
        path = os.path.join(cache, member_key(spec, prefix))
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    v = np.asarray(_compute_member(spec, prefix, root), np.float32)
    if cache:
        os.makedirs(cache, exist_ok=True)
        np.save(os.path.join(cache, member_key(spec, prefix)), v)
    return v


def _compute_member(spec, prefix, root=None):
    """spec: one config member. Returns [N] values, one per video/candidate."""
    win = spec.get("window", "full")
    if win == "chg":
        # self-change: the member on the first and on the last half of the clip, |b - a| / (|a| + |b|)
        a = _compute_member(dict(spec, window="first"), prefix, root)
        b = _compute_member(dict(spec, window="last"), prefix, root)
        return np.abs(b - a) / (np.abs(a) + np.abs(b) + 1e-12)
    store = open_store(prefix, spec["backbone"], spec["prep"], root)
    layer, sig, agg = spec["layer"], spec["signal"], spec["agg"]

    if sig in ("drift_motion", "stillness"):
        # drift_motion: mean distance of the pooled trajectory from its start, over the
        # median token step size; stillness: the median token step size over that of the
        # 10 stillest tokens
        sq, _ = _token_steps(store, layer, spec.get("tvar", "raw"), win)
        sp = np.nanmean(np.sqrt(np.clip(sq, 0, None)), -1)
        med = np.nanmedian(sp, 1)
        if sig == "stillness":
            return np.asarray(med / (np.nanmean(np.sort(sp, 1)[:, :10], 1) + 1e-9), np.float64)
        drift = gram_signals(_pooled_gram(store, layer, agg, spec.get("norm", "-"), win), ars=[])["dstart"]
        return np.asarray(drift / (med + 1e-9), np.float64)

    if agg == "tok":
        sq, dt = _token_steps(store, layer, spec.get("tvar", "raw"), win)
        per_token = token_signals(sq, dt)
        if sig not in per_token:
            raise KeyError("signal %s has no token form" % sig)
        return np.asarray(token_reduce(per_token[sig])[spec.get("post", "all")], np.float64)

    out = gram_signals(_pooled_gram(store, layer, agg, spec.get("norm", "-"), win))
    if sig not in out:
        raise KeyError("signal %s not produced for pooled path" % sig)
    return np.asarray(out[sig], np.float64)


# ------------------------------------------------------------- task adapters --
def detection_members(config, dataset, root=None, progress=True, cache=None):
    """Returns (list of [N_pairs] signed deltas, group labels).
    A delta is member(implausible) - member(plausible): positive when the member
    ranks the pair correctly."""
    block = config["datasets"][dataset]
    m0 = block["members"][0]
    pairs = open_store(dataset, m0["backbone"], m0["prep"], root).meta["pairs"]
    vi = np.array([p["vi"] for p in pairs])
    ii = np.array([p["ii"] for p in pairs])
    group = np.array([p.get("group", 0) for p in pairs])
    deltas = []
    n_total = len(block["members"])
    for k, spec in enumerate(block["members"], 1):
        if progress:
            print("  [%3d/%3d] %-8s %-12s %-6s" % (k, n_total, spec["backbone"],
                  spec["signal"], spec["agg"]), end="\r", flush=True)
        v = compute_member(spec, dataset, root, cache)
        deltas.append(v[ii] - v[vi])
    if progress:
        print("  computed %d members%s" % (n_total, " " * 40), flush=True)
    return deltas, group


def verification_members(config, gt_csv, root=None, progress=True, cache=None):
    """Returns a list of [N_scenarios, 16] member arrays, aligned to the GT table."""
    import pandas as pd
    G = pd.read_csv(gt_csv)
    scen = list(G["scenario_id"])
    sidx = {s: i for i, s in enumerate(scen)}
    NS, NC = len(scen), 16
    out = []
    n_total = len(config["members"])
    for k, spec in enumerate(config["members"], 1):
        if progress:
            print("  [%2d/%2d] %-8s %-10s %-6s %-6s" % (k, n_total, spec["backbone"],
                  spec["signal"], spec["agg"], spec.get("window", "full")),
                  end="\r", flush=True)
        M = np.full((NS, NC), np.nan)
        # the candidates sit in one store (v2v) or are split over two (v2v, v2v16);
        # every store records each video's scenario and absolute candidate index
        for prefix in ("v2v", "v2v16"):
            try:
                st = open_store(prefix, spec["backbone"], spec["prep"], root)
            except FileNotFoundError:
                continue
            v = compute_member(spec, prefix, root, cache)
            for e in st.meta["index"]:
                c = int(e["cand"])
                if e["scen"] in sidx and 0 <= c < NC:
                    M[sidx[e["scen"]], c] = v[e["vi"]]
        out.append(M)
    if progress:
        print("  computed %d members%s" % (n_total, " " * 40), flush=True)
    return out
