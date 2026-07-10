import random

import numpy as np
import torch


def set_seed(seed):
    print(f"SEED ID: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_rows(X, eps=1e-8):
    X = torch.as_tensor(X, dtype=torch.float32)
    return X / X.norm(p=2, dim=1, keepdim=True).clamp_min(float(eps))


def make_sites(layer, token_position, total_dim, resolution):
    sites = []
    for start in range(0, int(total_dim), int(resolution)):
        end = min(start + int(resolution), int(total_dim))
        sites.append((int(layer), token_position, int(start), int(end)))
    return sites


def compute_iia(outputs, labels, var_name, pointer_num_labels=4):
    scores = torch.as_tensor(outputs)
    labels = torch.as_tensor(labels, dtype=torch.long)

    if var_name == "answer_pointer":
        scores = scores[:, :pointer_num_labels]
    elif var_name != "answer_token":
        raise ValueError(f"unknown var_name={var_name!r}")

    pred = scores.argmax(dim=-1).cpu()
    labels = labels.cpu()
    correct = int((pred == labels).sum().item())
    return correct / int(labels.numel()), correct
