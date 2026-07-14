import torch
import random
import numpy as np
from itertools import combinations


def set_seed(seed):
    print(f"SEED ID: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def normalize_rows(X, eps=1e-8):
    X = torch.as_tensor(
        X,
        dtype=torch.float32,
    )

    row_norms = X.norm(
        p=2,
        dim=1,
        keepdim=True,
    )

    return X / row_norms.clamp_min(
        float(eps)
    )


def make_site_combination_candidates(sites, selected_indices, config_size):
    """Create all site configurations with config_size sites."""
    config_sites = []
    config_indices = []

    for indices in combinations(selected_indices, config_size):
        current_sites = []

        for i in indices:
            current_sites.append(sites[i])

        config_sites.append(current_sites)
        config_indices.append(indices)

    return config_sites, config_indices