from scipy.spatial.distance import cdist
import numpy as np
import torch
import math

def solve_ot(G, S, eps=0.05, tau=1.0, metric="sqeuclidean", num_iters=10000, tol=1e-9):

    C = torch.as_tensor(
        cdist(np.asarray(G, dtype=np.float64), np.asarray(S, dtype=np.float64), metric=metric),
        dtype=torch.float64,
    )
    n, m = C.shape
    p = torch.full((n,), 1.0 / n, dtype=torch.float64)
    q = torch.full((m,), 1.0 / m, dtype=torch.float64)
    reg = eps * tau
    u, v = torch.zeros(n, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
    H = lambda u, v: (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / reg
    for _ in range(num_iters):
        u_old, v_old = u, v
        u = reg * (torch.log(p + 1e-8) - torch.logsumexp(H(u, v), dim=-1)) + u
        v = reg * (torch.log(q + 1e-8) - torch.logsumexp(H(u, v).transpose(-1, -2), dim=-1)) + v
        if ((u - u_old).abs().sum() + (v - v_old).abs().sum()).item() < tol:
            break
    return torch.exp(H(u, v))


def solve_uot(G, S, eps=0.05, reg_m=(float("inf"), 1.0), metric="sqeuclidean", num_iters=10000, tol=1e-9):

    C = torch.tensor(
        cdist(np.asarray(G, float), np.asarray(S, float), metric=metric),
        dtype=torch.float64,
    )

    n, m = C.shape
    p = torch.full((n,), 1 / n, dtype=torch.float64)
    q = torch.full((m,), 1 / m, dtype=torch.float64)

    beta_a, beta_n = reg_m
    lam_a = 1.0 if math.isinf(beta_a) else beta_a / (beta_a + eps)
    lam_n = 1.0 if math.isinf(beta_n) else beta_n / (beta_n + eps)

    u = torch.zeros(n, dtype=torch.float64)
    v = torch.zeros(m, dtype=torch.float64)

    def H():
        return (-C + u[:, None] + v[None, :]) / eps

    for _ in range(num_iters):
        u_old, v_old = u.clone(), v.clone()

        u = lam_a * (eps * (torch.log(p) - torch.logsumexp(H(), dim=1)) + u)
        v = lam_n * (eps * (torch.log(q) - torch.logsumexp(H(), dim=0)) + v)

        if ((u - u_old).abs().sum() + (v - v_old).abs().sum()) < tol:
            break

    return torch.exp(H())