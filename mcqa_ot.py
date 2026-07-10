import math

import numpy as np
import torch
from scipy.spatial.distance import cdist


def solve_ot(G, S, eps=0.05, tau=1.0, metric="sqeuclidean", num_iters=10000, tol=1e-9):
    C = torch.as_tensor(cdist(np.asarray(G, dtype=np.float64), np.asarray(S, dtype=np.float64), metric=metric), dtype=torch.float64)
    n, m = C.shape
    p = torch.full((n,), 1.0 / n, dtype=torch.float64)
    q = torch.full((m,), 1.0 / m, dtype=torch.float64)
    reg = eps * tau
    u = torch.zeros(n, dtype=torch.float64)
    v = torch.zeros(m, dtype=torch.float64)

    def H():
        return (-C + u[:, None] + v[None, :]) / reg

    for _ in range(num_iters):
        u_old = u.clone()
        v_old = v.clone()
        u = reg * (torch.log(p + 1e-8) - torch.logsumexp(H(), dim=1)) + u
        v = reg * (torch.log(q + 1e-8) - torch.logsumexp(H(), dim=0)) + v

        if ((u - u_old).abs().sum() + (v - v_old).abs().sum()).item() < tol:
            break

    return torch.exp(H())


def solve_uot(G, S, eps=0.05, reg_m=(float("inf"), 1.0), metric="sqeuclidean", num_iters=10000, tol=1e-9):
    C = torch.as_tensor(cdist(np.asarray(G, dtype=np.float64), np.asarray(S, dtype=np.float64), metric=metric), dtype=torch.float64)
    n, m = C.shape
    p = torch.full((n,), 1.0 / n, dtype=torch.float64)
    q = torch.full((m,), 1.0 / m, dtype=torch.float64)

    beta_a, beta_n = reg_m
    lam_a = 1.0 if math.isinf(beta_a) else beta_a / (beta_a + eps)
    lam_n = 1.0 if math.isinf(beta_n) else beta_n / (beta_n + eps)

    u = torch.zeros(n, dtype=torch.float64)
    v = torch.zeros(m, dtype=torch.float64)

    def H():
        return (-C + u[:, None] + v[None, :]) / eps

    for _ in range(num_iters):
        u_old = u.clone()
        v_old = v.clone()
        u = lam_a * (eps * (torch.log(p) - torch.logsumexp(H(), dim=1)) + u)
        v = lam_n * (eps * (torch.log(q) - torch.logsumexp(H(), dim=0)) + v)

        if ((u - u_old).abs().sum() + (v - v_old).abs().sum()).item() < tol:
            break

    return torch.exp(H())


def get_solver(name):
    if name == "ot":
        return solve_ot
    if name == "uot":
        return solve_uot
    raise ValueError(f"unknown solver={name!r}")


def top_sites_from_T(T, sites, var_id, top_k, min_mass=1e-8):
    valid = []
    for i in range(len(sites)):
        value = T[var_id, i]
        if bool(torch.isfinite(value).item()) and float(value.detach().cpu().item()) > min_mass:
            valid.append(i)

    if not valid:
        raise ValueError(f"no positive-mass sites for var_id={var_id}")

    scores = []
    for i in valid:
        scores.append(T[var_id, i])
    scores = torch.stack(scores)

    k = min(int(top_k), len(valid))
    top = torch.topk(scores, k=k).indices.tolist()

    selected_sites = []
    selected_indices = []
    for local_i in top:
        global_i = valid[int(local_i)]
        selected_sites.append(sites[global_i])
        selected_indices.append(global_i)

    return selected_sites, selected_indices
