"""
Geometric signals computed from frame-trajectory Gram matrices.

A video is encoded by a frozen backbone into one feature vector per frame.
The Gram matrix G[t,s] = <z_t, z_s> of those vectors determines every distance,
angle and curvature along the trajectory without ever materialising the features,
which is why all signals below are functions of G alone.

Two families:
  gram_signals   -- pooled trajectory: one trajectory per video (whole-frame feature)
  token_signals  -- per-token trajectories: one trajectory per spatial patch,
                    later reduced across tokens by token_reduce
"""
import warnings
import numpy as np

# tokens or windows with too few valid steps give NaN, and the member then abstains;
# numpy warns about each such empty slice, which is expected here
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)

EPS = 1e-9
ARS = [3]          # autoregressive orders computed by default


def gram_signals(Gm, ars=ARS):
    """Gm [N,T,T] NaN-padded Gram matrices -> dict of signal name -> [N] scalars."""
    out = {}
    Gv = Gm[:, 1:, 1:] - Gm[:, 1:, :-1] - Gm[:, :-1, 1:] + Gm[:, :-1, :-1]
    sq = np.diagonal(Gv, axis1=-2, axis2=-1).copy()            # squared step norms
    dot = np.diagonal(Gv, offset=1, axis1=-2, axis2=-1).copy()  # consecutive step inner products
    s = np.sqrt(np.clip(sq, 0, None))                           # step lengths
    cos = dot / (s[:, :-1] * s[:, 1:] + EPS)
    ang = np.arccos(np.clip(cos, -1 + 1e-7, 1 - 1e-7))
    a2 = sq[:, 1:] - 2 * dot + sq[:, :-1]                       # squared acceleration
    an = np.sqrt(np.clip(a2, 0, None))
    m, sd = np.nanmean, np.nanstd

    out["curv"] = m(ang, -1)
    out["angcons"] = sd(ang, -1)
    out["speedstd"] = sd(s, -1)
    out["speedcv"] = sd(s, -1) / (m(s, -1) + EPS)
    out["accel"] = m(a2, -1)
    out["prederr"] = m(an / (s[:, :-1] + EPS), -1)

    N = Gm.shape[0]

    # displacement from the first frame
    d0 = np.sqrt(np.clip(
        np.diagonal(Gm, axis1=-2, axis2=-1) - 2 * Gm[:, 0, :] + Gm[:, 0, 0][:, None],
        0, None))
    out["dstart"] = np.nanmean(d0, -1)

    # return: how far the trajectory ends from where it began (1 - cosine between the
    # mean of its first 5 and of its last 5 frames)
    ret = np.full(N, np.nan)
    for i in range(N):
        g = Gm[i]
        v = np.where(~np.isnan(np.diagonal(g)))[0]
        if len(v) < 20:
            continue
        a, b = v[:5], v[-5:]
        ret[i] = 1.0 - g[np.ix_(a, b)].mean() / np.sqrt(g[np.ix_(a, a)].mean() * g[np.ix_(b, b)].mean() + 1e-12)
    out["return"] = ret

    # jerk: third differences along the trajectory
    jerk = an[:, 1:] - an[:, :-1]
    out["jerk"] = np.nanmean(np.abs(jerk), -1)

    # participation ratio of the centred Gram (effective dimensionality)
    pr = np.full(N, np.nan)
    for i in range(N):
        g = Gm[i]
        d = np.diagonal(g)
        v = np.where(~np.isnan(d))[0]
        if len(v) < 4:
            continue
        g = g[np.ix_(v, v)]
        n = g.shape[0]
        H = np.eye(n) - 1.0 / n
        ev = np.clip(np.linalg.eigvalsh(H @ g @ H), 0, None)
        pr[i] = (ev.sum() ** 2) / (np.sum(ev ** 2) + EPS)
    out["partratio"] = pr

    # autoregressive residual: how well frame t+1 is predicted by the previous H frames
    for H in ars:
        res = np.full(N, np.nan)
        for i in range(N):
            g = Gm[i]
            d = np.diagonal(g)
            v = np.where(~np.isnan(d))[0]
            t = len(v)
            if t < H + 3:
                continue
            g = np.nan_to_num(g[np.ix_(v, v)])
            A = np.zeros((H, H)); b = np.zeros(H); c = 0.0; n = 0
            for u in range(H, t - 1):
                hi = np.arange(u - H + 1, u + 1)
                A += g[np.ix_(hi, hi)]; b += g[hi, u + 1]; c += g[u + 1, u + 1]; n += 1
            try:
                al = np.linalg.solve(A + 1e-6 * np.trace(A) / H * np.eye(H), b)
            except np.linalg.LinAlgError:
                continue
            res[i] = max((c - 2 * b @ al + al @ A @ al) / n, 0.0)
        out["ar%d" % H] = res
    return out


def token_signals(sq, dot):
    """sq [N,P,T-1], dot [N,P,T-2] -> dict signal -> [N,P] (one value per token)."""
    s = np.sqrt(np.clip(sq, 0, None))
    cos = dot / (s[..., :-1] * s[..., 1:] + EPS)
    ang = np.arccos(np.clip(cos, -1 + 1e-7, 1 - 1e-7))
    a2 = sq[..., 1:] - 2 * dot + sq[..., :-1]
    an = np.sqrt(np.clip(a2, 0, None))
    m, sd = np.nanmean, np.nanstd
    jerk = an[..., 1:] - an[..., :-1]
    return {
        "curv": m(ang, -1),
        "angcons": sd(ang, -1),
        "speedstd": sd(s, -1),
        "speedcv": sd(s, -1) / (m(s, -1) + EPS),
        "accel": m(a2, -1),
        "prederr": m(an / (s[..., :-1] + EPS), -1),
        "jerk": m(np.abs(jerk), -1),
        "dstart": m(np.cumsum(np.nan_to_num(s), -1), -1),
    }


def token_reduce(V):
    """V [N,P] per-token values -> dict rule name -> [N]. Cross-token aggregations."""
    o = {}
    order = np.sort(V, axis=1)
    o["all"] = np.nanmean(V, 1)
    for k in (5, 10, 25, 50, 100):
        if k <= V.shape[1]:
            o["bottom%d" % k] = np.nanmean(order[:, :k], 1)
            o["top%d" % k] = np.nanmean(order[:, -k:], 1)
    for q in (10, 25):
        lo = np.nanpercentile(V, q, axis=1)
        hi = np.nanpercentile(V, 100 - q, axis=1)
        o["band%d" % q] = np.array([
            np.nanmean(V[i][(V[i] >= lo[i]) & (V[i] <= hi[i])]) for i in range(V.shape[0])])
    o["median"] = np.nanmedian(V, 1)
    o["iqr"] = np.nanpercentile(V, 75, axis=1) - np.nanpercentile(V, 25, axis=1)
    o["max"] = np.nanmax(V, 1)
    o["min"] = np.nanmin(V, 1)
    o["std"] = np.nanstd(V, 1)
    o["cv"] = np.nanstd(V, 1) / (np.nanmean(V, 1) + EPS)
    w = np.nan_to_num(V)
    o["wmean"] = (w * w).sum(1) / (w.sum(1) + EPS)
    return o

