"""
Ensembling: turning per-member values into a decision.

A member produces one scalar per item (per candidate for verification, per pair for
detection). Members live on incomparable scales, so each is normalised before being
combined. Nothing here is learned: the normalisation and the rule are fixed.
"""
import numpy as np

EPS = 1e-9


# ------------------------------------------------- verification (best of N) --
def z_scale(X):
    """Standardise within each row (one scenario's candidates). X [N,K] -> [N,K]."""
    mu = np.nanmean(X, 1, keepdims=True)
    sd = np.nanstd(X, 1, keepdims=True) + EPS
    return np.nan_to_num(np.clip((X - mu) / sd, -30, 30))


def z_mad(X):
    """Median/MAD-robust standardisation within each row."""
    med = np.nanmedian(X, 1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(X - med), 1, keepdims=True) + EPS
    return np.nan_to_num(np.clip((X - med) / mad, -30, 30))


def z_rank(X):
    """Rank within each row, scaled to [0,1]."""
    K = X.shape[1]
    x = np.where(np.isnan(X), 9e18, X)
    return (np.argsort(np.argsort(x, 1), 1) / max(K - 1, 1)).astype(np.float32)


# fusion rule -> per-scenario normalisation; the normalised members are then summed
FUSION = {"rank": z_rank, "sumz_scale": z_scale, "sumz_mad": z_mad}


def fuse(members, rule):
    """members: list of [N_scenarios, N_candidates] values (lower = more plausible).
    Returns the fused [N, K] scores; the argmin over candidates is the selection."""
    f = FUSION[rule]
    return np.stack([f(np.asarray(X, np.float64)) for X in members]).sum(0)


# --------------------------------------------------------- detection (pairs) --
def by_cell(Z, cells):
    """Z [members, pairs]: each member's scaled delta (implausible minus plausible).
    Each signal-encoder cell votes with its members' signs (ties broken by their sum),
    then the cells vote (ties broken by the sum over all members)."""
    cells = np.asarray(cells); V = np.zeros(Z.shape[1])
    for u in np.unique(cells):
        z = Z[cells == u]; v = np.sign(z).sum(0)
        V += np.where(np.where(v == 0, z.sum(0) > 0, v > 0), 1.0, -1.0)
    return np.where(V == 0, Z.sum(0), V)


def vote_pairs(deltas, specs):
    """deltas [members, pairs], specs: the configuration's members (each with its scale).
    Returns the vote per pair: positive when the second video is judged implausible."""
    D = np.asarray(deltas, np.float64)
    Z = np.nan_to_num(D / (np.array([m["scale"] for m in specs], np.float64)[:, None] + EPS))
    return by_cell(Z, ["%s|%s" % (m["signal"], m["backbone"]) for m in specs])


def macro_micro(correct, group):
    """Macro = mean over scenario groups; micro = mean over pairs."""
    accs = [correct[group == u].mean() for u in np.unique(group) if (group == u).any()]
    return 100.0 * float(np.mean(accs)), 100.0 * float(correct.mean())
