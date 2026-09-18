import argparse
import os
import random
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mcqa_data_load_all import build_mcqa_banks, letter_token_id
from mcqa_neural_net import load_gemma_model


ANSWER_LETTERS = tuple(string.ascii_uppercase)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def answer_label_ids(tokenizer):
    return [letter_token_id(tokenizer, letter) for letter in ANSWER_LETTERS]


def answer_logits(model, outputs, attention_mask, label_ids):
    device = outputs.last_hidden_state.device
    rows = torch.arange(attention_mask.shape[0], device=device)
    idx = torch.arange(attention_mask.shape[1], device=device)
    pos = (attention_mask * idx.unsqueeze(0)).max(dim=1).values

    hidden = outputs.last_hidden_state[rows, pos, :]
    ids = torch.tensor(label_ids, device=device)
    W = model.lm_head.weight[ids].to(hidden.dtype)
    logits = hidden @ W.T

    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[ids]

    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits.float()


def ap_iia(logits, target):
    pred = logits[:, :4].argmax(dim=-1).cpu()
    return float((pred == target.cpu()).float().mean())


def at_readout_accuracy(logits, target):
    pred = logits.argmax(dim=-1).cpu()
    return float((pred == target.cpu()).float().mean())


class LearnedSubspace(nn.Module):
    def __init__(self, hidden_size, subspace_dim):
        super().__init__()
        init = torch.randn(hidden_size, subspace_dim)
        self.raw = nn.Parameter(torch.linalg.qr(init, mode="reduced").Q)

    def basis(self):
        return torch.linalg.qr(self.raw, mode="reduced").Q


@torch.no_grad()
def collect_layer_states(model, input_ids, attention_mask, position_by_id, layer, token_position="last_token", batch_size=32):
    device = next(model.parameters()).device
    states = []

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + position_by_id[token_position][start:end].to(device)
        captured = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["x"] = hidden[rows, pos, :].detach().float().cpu()

        handle = model.model.layers[layer].register_forward_hook(hook)
        model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
        handle.remove()

        states.append(captured["x"])

    return torch.cat(states)


@torch.no_grad()
def collect_all_layer_states(model, input_ids, attention_mask, position_by_id, layers, token_position="last_token", batch_size=32):
    return {layer: collect_layer_states(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size) for layer in layers}


@torch.no_grad()
def full_layer_ap_iia(model, bank, source_states, layer, label_ids, token_position="last_token", batch_size=16):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_states[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            hidden_new[rows, pos, :] = source.to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        handle = model.model.layers[layer].register_forward_hook(hook)
        outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
        handle.remove()

        all_logits.append(answer_logits(model, outputs, mask, label_ids).cpu())

    logits = torch.cat(all_logits)
    return ap_iia(logits, bank["counterfactual_label_ids"]["answer_pointer"])


def run_ap_intervention_and_capture_at(model, bank, source_ap_states, Q_ap, ap_layer, at_layer, label_ids, token_position="last_token", batch_size=8, require_grad=True):
    device = next(model.parameters()).device
    all_logits = []
    all_generated_at = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_ap_states[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)
        captured = {}

        def ap_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()

            base = hidden[rows, pos, :].float()
            delta = source.float() - base
            hidden_new[rows, pos, :] = (base + (delta @ Q_ap) @ Q_ap.T).to(hidden.dtype)

            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        def at_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["at"] = hidden[rows, pos, :].float()

        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        at_handle = model.model.layers[at_layer].register_forward_hook(at_hook)

        outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)

        ap_handle.remove()
        at_handle.remove()

        logits = answer_logits(model, outputs, mask, label_ids)
        generated_at = captured["at"]

        if require_grad:
            all_logits.append(logits)
            all_generated_at.append(generated_at)
        else:
            all_logits.append(logits.detach().cpu())
            all_generated_at.append(generated_at.detach().cpu())

    return torch.cat(all_logits), torch.cat(all_generated_at)


def at_readout_logits(generated_at, Q_at, W_at, b_at):
    return (generated_at @ Q_at) @ W_at.T + b_at


@torch.no_grad()
def evaluate_ap(model, bank, source_ap_states, Q_ap, ap_layer, at_layer, Q_at, W_at, b_at, label_ids, token_position="last_token", batch_size=16):
    direct_logits, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap_states, Q_ap, ap_layer, at_layer, label_ids, token_position, batch_size, require_grad=False)

    generated_at = generated_at.to(Q_at.device)
    readout_logits = at_readout_logits(generated_at, Q_at, W_at, b_at)

    direct_iia = ap_iia(direct_logits, bank["counterfactual_label_ids"]["answer_pointer"])
    readout_acc = at_readout_accuracy(readout_logits, bank["counterfactual_label_ids"]["answer_token"])

    return direct_iia, readout_acc


def make_mini_bank(bank, idx):
    return {
        "base_input_ids": bank["base_input_ids"][idx],
        "base_attention_mask": bank["base_attention_mask"][idx],
        "base_position_by_id": {k: v[idx] for k, v in bank["base_position_by_id"].items()}
    }


def train_phase2_ap(model, tokenizer, at_checkpoint, at_readout_checkpoint, layers=None, subspace_dim=128, ft_size=None, cal_size=None, te_size=None, epochs=50, lr=1e-2, lambda_readout=1.0, train_batch_size=32, eval_batch_size=32, token_position=None, seed=0, save_path="results/gradual_das_ap_readout.pt"):
    set_seed(seed)
    device = next(model.parameters()).device

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    labels = answer_label_ids(tokenizer)

    at_result = torch.load(at_checkpoint, map_location="cpu")
    at_layer = int(at_result["layer"])
    Q_at = at_result["basis"].float().to(device)

    readout_result = torch.load(at_readout_checkpoint, map_location="cpu")
    W_at = readout_result["W_AT"].float().to(device)
    b_at = readout_result["b_AT"].float().to(device)

    if token_position is None:
        token_position = at_result["token_position"]

    config = at_result["config"]
    ft_size = config["ft_size"] if ft_size is None else ft_size
    cal_size = config["cal_size"] if cal_size is None else cal_size
    te_size = config["te_size"] if te_size is None else te_size

    fit_bank, cal_banks, te_banks = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=ft_size, cal_size=cal_size, te_size=te_size, dataset_size=None, split="train", device=device, batch_size=eval_batch_size, seed=seed)
    cal_bank = cal_banks["answer_pointer"]
    te_bank = te_banks["answer_pointer"]

    if layers is None:
        layers = list(range(at_layer))

    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)

    layer_results = []
    for layer in layers:
        iia = full_layer_ap_iia(model, cal_bank, cal_source_by_layer[layer], layer, labels, token_position, eval_batch_size)
        layer_results.append({"layer": layer, "cal_iia": iia})
        print(f"[AP layer] layer={layer} iia={iia:.4f}")

    ap_layer = max(layer_results, key=lambda x: x["cal_iia"])["layer"]
    print(f"[AP layer selected] {ap_layer}")

    ft_source_ap = collect_layer_states(model, fit_bank["source_input_ids"], fit_bank["source_attention_mask"], fit_bank["source_position_by_id"], ap_layer, token_position, eval_batch_size)
    cal_source_ap = cal_source_by_layer[ap_layer]
    te_source_ap = collect_layer_states(model, te_bank["source_input_ids"], te_bank["source_attention_mask"], te_bank["source_position_by_id"], ap_layer, token_position, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    target_ap_ft = fit_bank["counterfactual_label_ids"]["answer_pointer"]
    target_at_ft = fit_bank["counterfactual_label_ids"]["answer_token"]

    best_score = -1.0
    best_basis = None
    best_epoch = None
    best_direct_cal = None
    best_readout_cal = None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(fit_bank["base_input_ids"]))
        total_direct_loss = 0.0
        total_readout_loss = 0.0
        total_count = 0

        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]
            mini_bank = make_mini_bank(fit_bank, idx)
            mini_source_ap = ft_source_ap[idx]

            target_ap = target_ap_ft[idx].to(device)
            target_at = target_at_ft[idx].to(device)

            optimizer.zero_grad()
            Q_ap = alignment.basis()

            direct_logits, generated_at = run_ap_intervention_and_capture_at(model, mini_bank, mini_source_ap, Q_ap, ap_layer, at_layer, labels, token_position, len(idx), require_grad=True)

            direct_loss = F.cross_entropy(direct_logits[:, :4], target_ap)
            readout_loss = F.cross_entropy(at_readout_logits(generated_at, Q_at, W_at, b_at), target_at)
            loss = direct_loss + lambda_readout * readout_loss

            loss.backward()
            optimizer.step()

            n = len(idx)
            total_direct_loss += direct_loss.item() * n
            total_readout_loss += readout_loss.item() * n
            total_count += n

        Q_eval = alignment.basis()
        direct_cal, readout_cal = evaluate_ap(model, cal_bank, cal_source_ap, Q_eval, ap_layer, at_layer, Q_at, W_at, b_at, labels, token_position, eval_batch_size)
        score = direct_cal if lambda_readout == 0 else min(direct_cal, readout_cal)

        print(f"[epoch {epoch:02d}] direct_loss={total_direct_loss / total_count:.4f} readout_loss={total_readout_loss / total_count:.4f} direct_cal={direct_cal:.4f} readout_cal={readout_cal:.4f} score={score:.4f}")

        if score > best_score:
            best_score = score
            best_basis = Q_eval.detach().cpu()
            best_epoch = epoch
            best_direct_cal = direct_cal
            best_readout_cal = readout_cal

    Q_ap = best_basis.to(device)
    direct_test, readout_test = evaluate_ap(model, te_bank, te_source_ap, Q_ap, ap_layer, at_layer, Q_at, W_at, b_at, labels, token_position, eval_batch_size)

    result = {
        "variable": "answer_pointer",
        "method": "AP direct loss + frozen AT readout loss",
        "ap_layer": ap_layer,
        "at_layer": at_layer,
        "ap_basis": best_basis,
        "at_checkpoint": at_checkpoint,
        "at_readout_checkpoint": at_readout_checkpoint,
        "lambda_readout": lambda_readout,
        "cal": {"best_epoch": best_epoch, "direct_iia": best_direct_cal, "readout_accuracy": best_readout_cal, "score": best_score},
        "test": {"direct_iia": direct_test, "readout_accuracy": readout_test},
        "token_position": token_position,
        "config": {"ft_size": ft_size, "cal_size": cal_size, "te_size": te_size, "epochs": epochs, "lr": lr, "lambda_readout": lambda_readout, "train_batch_size": train_batch_size, "eval_batch_size": eval_batch_size, "seed": seed}
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)

    print(f"[AP BEST] layer={ap_layer} epoch={best_epoch} direct_cal={best_direct_cal:.4f} readout_cal={best_readout_cal:.4f}")
    print(f"[AP TEST] direct_iia={direct_test:.4f} readout_accuracy={readout_test:.4f}")
    print(f"saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at-checkpoint", default="results/gradual_das_at_layer_selected.pt")
    parser.add_argument("--at-readout-checkpoint", default="results/at_readout_LogisticRegression.pt")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--subspace-dim", type=int, default=128)
    parser.add_argument("--ft-size", type=int, default=None)
    parser.add_argument("--cal-size", type=int, default=None)
    parser.add_argument("--te-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-readout", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--token-position", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="results/gradual_das_ap_readout.pt")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    train_phase2_ap(model, tokenizer, args.at_checkpoint, args.at_readout_checkpoint, layers, args.subspace_dim, args.ft_size, args.cal_size, args.te_size, args.epochs, args.lr, args.lambda_readout, args.train_batch_size, args.eval_batch_size, args.token_position, args.seed, args.save_path)


if __name__ == "__main__":
    main()