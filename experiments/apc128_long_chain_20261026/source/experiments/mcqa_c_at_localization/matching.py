"""Canonical effect signatures and converged, one-sided entropic UOT."""
import numpy as np
from scipy.special import logsumexp


def signature_cost(abstract, neural):
    abstract, neural = np.asarray(abstract, float), np.asarray(neural, float)
    if abstract.ndim != 3 or neural.ndim != 3 or abstract.shape[1:] != neural.shape[1:]:
        raise ValueError("Signatures must have shape (variables/sites, pairs, four labels)")
    if abstract.shape[2] != 4 or not np.isfinite(abstract).all() or not np.isfinite(neural).all():
        raise ValueError("Invalid canonical signatures")
    return ((abstract[:, None] - neural[None, :]) ** 2).sum(axis=(2, 3)) / abstract.shape[1]


def one_sided_uot(cost, epsilon=0.05, beta=0.5, tolerance=1e-10, max_iter=10000):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or not cost.size or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("Invalid cost matrix")
    if epsilon <= 0 or beta <= 0:
        raise ValueError("Positive regularization required")
    loga = np.full(cost.shape[0], -np.log(cost.shape[0]))
    logb = np.full(cost.shape[1], -np.log(cost.shape[1]))
    kernel = -cost / epsilon
    logv = np.zeros(cost.shape[1])
    rho = beta / (beta + epsilon)
    for iteration in range(1, max_iter + 1):
        logu = loga - logsumexp(kernel + logv[None, :], axis=1)
        newv = rho * (logb - logsumexp(kernel + logu[:, None], axis=0))
        residual = float(np.max(np.abs(newv - logv)))
        logv = newv
        if residual <= tolerance:
            logu = loga - logsumexp(kernel + logv[None, :], axis=1)
            coupling = np.exp(kernel + logu[:, None] + logv[None, :])
            row_error = float(np.max(np.abs(coupling.sum(1) - np.exp(loga))))
            fixed = rho * (logb - logsumexp(kernel + logu[:, None], axis=0))
            fixed_error = float(np.max(np.abs(fixed - logv)))
            if row_error > tolerance or fixed_error > tolerance:
                continue
            return coupling, {"iterations": iteration, "row_error": row_error,
                              "fixed_point_error": fixed_error, "epsilon": epsilon, "beta": beta}
    raise RuntimeError("One-sided UOT did not converge; do not rank its unconverged coupling")


def rank_sites(sites, scores, reciprocal_ranks=None):
    if len(sites) != len(scores):
        raise ValueError("Site/score mismatch")
    ranks = reciprocal_ranks if reciprocal_ranks is not None else [0.] * len(sites)
    return sorted(range(len(sites)), key=lambda i: (-float(scores[i]), -float(ranks[i]),
                                                  sites[i]["layer"], sites[i]["position"]))


def select_winner(candidates):
    if not candidates:
        raise ValueError("Empty candidate set")
    return min(candidates, key=lambda c: (-c["cal_accuracy"], c["rank"], -c["epoch"],
                                          c["layer"], c["position"]))


def select_chain(candidates):
    cs = [c for c in candidates if c["variable"] == "C"]
    ats = [c for c in candidates if c["variable"] == "AT"]
    eligible = [(c, at) for c in cs for at in ats
                if c["layer"] < at["layer"] and c["position"] <= at["position"]]
    if not eligible:
        return None
    def key(pair):
        c, at = pair
        return (-min(c["cal_accuracy"], at["cal_accuracy"]),
                -(c["cal_accuracy"] + at["cal_accuracy"]) / 2,
                c["rank"] + at["rank"], c["layer"], c["position"], at["layer"], at["position"],
                c["rank"], at["rank"], -c["epoch"], -at["epoch"])
    c, at = min(eligible, key=key)
    return {"C": c, "AT": at}
