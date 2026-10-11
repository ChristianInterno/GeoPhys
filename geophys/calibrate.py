"""
Calibrate GeoPhys on your own videos.

    my_pairs/
      case_a/
        a_plausible.mp4      in each case the plausible video must sort first
        b_implausible.mp4
      case_b/
        ...

    python -m geophys.calibrate --pairs my_pairs/ --out configs/mine.json
    python -m geophys.pairs --videos new_pairs/ --config configs/mine.json

Two ways to choose the configuration (--metric):

  curvature   (default) perceptual straightening: physically implausible motion bends the
              trajectory of frame features. For each encoder, the layer where the implausible
              video of a pair is most curved relative to the plausible one (mean curvature
              difference over its standard deviation); at that layer, the --size measurements
              with the largest standardised difference between the two videos.
  accuracy    at the usual readout layer of each encoder, measurements are added one at a
              time, each time the one that most improves the accuracy on your pairs.

The command also runs the whole procedure on half of the pairs
and tests on the other half (5 times) and reports that held-out accuracy: trust it.
"""
import os
import json
import argparse

import numpy as np

from .backbones import READOUT, get_backbone, load_frames, prep_frames
from .extract import STORES, ensure_store, _forward
from .members import open_store, window_slice, _CACHE
from .signals import gram_signals, token_signals, token_reduce
from .ensemble import vote_pairs
from .pairs import index_pairs

WINDOWS = ["full", "mid", "first", "last"]
PREPS = ["direct", "hf"]


# --------------------------------------------------------------- vocabulary --
def _chg(a, b):
    return np.abs(b - a) / (np.abs(a) + np.abs(b) + 1e-12)


def store_values(st, layer):
    """Every measurement on one store, as {(signal, name): [N] values}. The values are
    those members.compute_member returns for the member spec_of builds from the name."""
    out, med = {}, {}
    li = [str(x) for x in st.layers].index(str(layer))
    for tv in ("raw", "l2", "spd"):                       # per-token trajectories
        SQ = np.array(st.tokens(tv)[0][:, li], np.float32)
        DT = np.array(st.tokens(tv)[1][:, li], np.float32)
        nf, per = SQ.shape[-1] + 1, {}
        for w in WINDOWS:
            fi = np.arange(nf)[window_slice(w, nf)]
            sq = SQ[:, :, fi[0]:fi[-1]].copy()
            dt = DT[:, :, fi[0]:max(fi[-1] - 1, fi[0] + 1)].copy()
            sq[sq == 0] = np.nan
            dt[np.isnan(sq[..., :-1])] = np.nan
            for s, V in token_signals(sq, dt).items():
                for rule, v in token_reduce(V).items():
                    per[(s, rule, w)] = out[(s, "tok/%s/%s/%s" % (tv, rule, w))] = np.asarray(v, np.float64)
            sp = np.nanmean(np.sqrt(np.clip(sq, 0, None)), -1)
            med[(tv, w)] = np.nanmedian(sp, 1)
            per[("stillness", "", w)] = out[("stillness", "tok/%s/%s" % (tv, w))] = \
                med[(tv, w)] / (np.nanmean(np.sort(sp, 1)[:, :10], 1) + 1e-9)
        for (s, rule, w) in [k for k in per if k[2] == "first"]:
            name = "tok/%s/chg" % tv if s == "stillness" else "tok/%s/%s/chg" % (tv, rule)
            out[(s, name)] = _chg(per[(s, rule, "first")], per[(s, rule, "last")])
    for atom in [a for a in ("mean", "cls", "flat") if a in st.atoms]:   # whole-frame trajectories
        G0 = np.array(st.gram[:, li, st.atoms.index(atom)], np.float64)
        G0[G0 == 0] = np.nan
        for norm in ("-", "l2"):
            G = G0
            if norm == "l2":
                d = np.sqrt(np.clip(np.diagonal(G0, axis1=1, axis2=2), 1e-9, None))
                G = G0 / (d[:, :, None] * d[:, None, :])
            per = {}
            for w in WINDOWS:
                sl = window_slice(w, G.shape[1])
                for s, v in gram_signals(np.ascontiguousarray(G[:, sl, sl])).items():
                    if s != "return" or (norm == "-" and w == "full"):
                        per[(s, w)] = out[(s, "%s/%s/%s" % (atom, norm, w))] = np.asarray(v, np.float64)
                for tv in ("raw", "l2"):
                    if atom != "flat":
                        per[("drift_motion", tv, w)] = out[("drift_motion", "%s/%s/%s/%s" % (atom, norm, tv, w))] = \
                            per[("dstart", w)] / (med[(tv, w)] + 1e-9)
            for k in [k for k in per if k[-1] == "first"]:
                if k[0] == "drift_motion":
                    out[(k[0], "%s/%s/%s/chg" % (atom, norm, k[1]))] = _chg(per[k], per[(k[0], k[1], "last")])
                elif k[0] != "return":
                    out[(k[0], "%s/%s/chg" % (atom, norm))] = _chg(per[k], per[(k[0], "last")])
    return out


def spec_of(backbone, prep, layer, signal, name):
    """A measurement as a configuration member."""
    p = name.split("/")
    m = dict(signal=signal, backbone=backbone, layer=str(layer), prep=prep,
             norm="-", tvar="-", post="-")
    if signal == "stillness":
        m.update(agg="tok", tvar=p[1], window=p[2])
    elif signal == "drift_motion":
        m.update(agg=p[0], norm=p[1], tvar=p[2], window=p[3])
    elif p[0] == "tok":
        m.update(agg="tok", tvar=p[1], post=p[2], window=p[3])
    else:
        m.update(agg=p[0], norm=p[1], window=p[2])
    return m


# ------------------------------------------------------------- the layer --
def layer_curvatures(tag, enc, vids, device):
    """{prep: {layer: [N] curvature of the mean-token trajectory}}, every layer of the
    encoder; cached next to the stores."""
    path = os.path.join(STORES, "%s_curvature_%s.json" % (tag, enc))
    if os.path.exists(path):
        c = json.load(open(path))
        if c["videos"] == list(vids):
            return {p: {L: np.array(v, np.float64) for L, v in d.items()} for p, d in c["curv"].items()}
    from tqdm import tqdm
    ext = get_backbone(enc, device)
    out = {p: {L: np.full(len(vids), np.nan) for L in ext.layers} for p in PREPS}
    print("  curvature at every layer of %s" % enc, flush=True)
    for i, v in enumerate(tqdm(vids)):
        frames = load_frames(v)
        if len(frames) < 4:
            continue
        for p in PREPS:
            pooled, _ = _forward(ext, prep_frames(frames, p), device)
            for L in ext.layers:
                z = pooled[L]["mean"].double()
                out[p][L][i] = gram_signals((z @ z.T).cpu().numpy()[None], ars=[])["curv"][0]
    os.makedirs(STORES, exist_ok=True)
    json.dump({"videos": list(vids), "curv": {p: {L: [None if np.isnan(x) else float(x) for x in a]
               for L, a in d.items()} for p, d in out.items()}}, open(path, "w"))
    return {p: {L: np.array(a, np.float64) for L, a in d.items()} for p, d in out.items()}


def gap(d):
    """standardised difference: mean over pairs of (implausible - plausible) over its spread"""
    return np.nanmean(d, -1) / (np.nanstd(d, -1) + 1e-12)


# ---------------------------------------------------------------- selection --
def _cell(S, Q):
    """a cell's vote from its members' sign sum S and value sum Q (as ensemble.by_cell)"""
    return np.where(np.where(S == 0, Q > 0, S > 0), 1.0, -1.0)


def forward(Z, cell, size):
    """Add members one at a time, each time the one giving the highest pair accuracy under
    the vote of ensemble.vote_pairs. Z [members, pairs] scaled deltas, cell [members]."""
    n = int(cell.max()) + 1
    S, Q, used = np.zeros((n, Z.shape[1])), np.zeros((n, Z.shape[1])), np.zeros(n, bool)
    V, Qall, free, chosen = np.zeros(Z.shape[1]), np.zeros(Z.shape[1]), np.ones(len(Z), bool), []
    for _ in range(min(size, len(Z))):
        best = (-1.0, -1)
        for c in range(n):
            j = np.where((cell == c) & free)[0]
            if len(j):
                V2 = V - (_cell(S[c], Q[c]) if used[c] else 0) + _cell(S[c] + np.sign(Z[j]), Q[c] + Z[j])
                acc = np.where(V2 == 0, Qall + Z[j] > 0, V2 > 0).mean(1)
                if acc.max() > best[0]:
                    best = (float(acc.max()), int(j[acc.argmax()]))
        j = best[1]; c = cell[j]
        V -= _cell(S[c], Q[c]) if used[c] else 0
        S[c] += np.sign(Z[j]); Q[c] += Z[j]; used[c] = True
        V += _cell(S[c], Q[c]); Qall += Z[j]; free[j] = False; chosen.append(j)
    return chosen


def main():
    ap = argparse.ArgumentParser(description="GeoPhys - calibrate a configuration on your own pairs")
    ap.add_argument("--pairs", required=True, help="directory of <case>/<two videos>, the plausible one first")
    ap.add_argument("--out", required=True, help="configuration file to write")
    ap.add_argument("--metric", choices=["curvature", "accuracy"], default="curvature")
    ap.add_argument("--size", type=int, default=16, help="number of members")
    ap.add_argument("--encoders", default=",".join(READOUT), help="comma-separated subset of %s" % ",".join(READOUT))
    ap.add_argument("--tag", default="calib", help="prefix for the extracted stores")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    pairs, vids = index_pairs(a.pairs)
    vi, ii = np.array([p["vi"] for p in pairs]), np.array([p["ii"] for p in pairs])
    meta = dict(pairs=[{"group": p["case"], "vi": p["vi"], "ii": p["ii"]} for p in pairs])
    encs = a.encoders.split(",")
    print("stores (%s):" % STORES)
    curv = {e: layer_curvatures(a.tag, e, vids, a.device) for e in encs} if a.metric == "curvature" else {}

    def layers(rows):
        """the layer of each encoder: the usual readout, or the largest curvature gap"""
        if a.metric == "accuracy":
            return {e: READOUT[e] for e in encs}
        out = {}
        for e in encs:
            g = {L: np.nan_to_num(np.mean([gap(curv[e][p][L][ii[rows]] - curv[e][p][L][vi[rows]]) for p in PREPS]), nan=-np.inf)
                 for L in curv[e][PREPS[0]]}
            out[e] = max(g, key=g.get)
        return out

    cache = {}

    def measurements(lay):
        """specs and deltas (implausible minus plausible) of every measurement at these layers"""
        specs, D = [], []
        for e, L in lay.items():
            for p in PREPS:
                if (e, p, L) not in cache:
                    prefix = "%s-L%s" % (a.tag, L)
                    ensure_store(prefix, e, p, vids, meta, [L], a.device)
                    _CACHE.clear()
                    vals = store_values(open_store(prefix, e, p), L)
                    cache[(e, p, L)] = ([spec_of(e, p, L, s, n) for s, n in sorted(vals)],
                                        np.array([vals[k][ii] - vals[k][vi] for k in sorted(vals)]))
                specs += cache[(e, p, L)][0]; D.append(cache[(e, p, L)][1])
        D = np.concatenate(D)
        keep = (np.isnan(D).mean(1) <= 0.1) & (np.nanstd(D, 1) > 0)
        specs = [dict(s, scale=float(np.sqrt(np.nanmean(d * d)))) for s, d, k in zip(specs, D, keep) if k]
        return specs, D[keep]                              # the scale uses no label

    def configure(rows):
        lay = layers(rows)
        specs, D = measurements(lay)
        if a.metric == "curvature":
            g = np.nan_to_num(gap(D[:, rows]), nan=-np.inf)
            chosen = list(np.argsort(-g, kind="stable")[:a.size])
        else:
            Z = np.nan_to_num(D / (np.array([s["scale"] for s in specs])[:, None] + 1e-9))
            cells = sorted({(s["signal"], s["backbone"]) for s in specs})
            cell = np.array([cells.index((s["signal"], s["backbone"])) for s in specs])
            chosen = forward(Z[:, rows], cell, a.size)
        return lay, [specs[j] for j in chosen], D[chosen]

    def accuracy(members, D, cols):
        return 100.0 * float((vote_pairs(D[:, cols], members) > 0).mean())

    allp = np.arange(len(pairs))
    lay, members, D = configure(allp)
    fit = accuracy(members, D, allp)
    held = []
    for seed in range(5):
        p = np.random.default_rng(seed).permutation(len(pairs))
        tr, te = np.sort(p[:len(p) // 2]), np.sort(p[len(p) // 2:])
        _, m_s, D_s = configure(tr)
        held.append(accuracy(m_s, D_s, te))

    print("%d pairs, metric %s; layers: %s" % (len(pairs), a.metric, ", ".join("%s %s" % kv for kv in lay.items())))
    cfg = {"method": os.path.splitext(os.path.basename(a.out))[0], "task": "detection",
           "datasets": {"custom": {"members": members}},
           "calibration": {"pairs": os.path.abspath(a.pairs), "metric": a.metric, "accuracy": round(fit, 2),
                           "held_out": round(float(np.mean(held)), 2)}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(cfg, open(a.out, "w"), indent=1)
    print("%d members; accuracy on these pairs %.2f%%, held-out %.2f%% (+- %.2f, chance 50)"
          % (len(members), fit, np.mean(held), np.std(held)))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
