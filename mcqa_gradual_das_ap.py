"""Train AP DAS and optionally enforce the downstream AT mediator/readout."""

import os

import torch
import torch.nn.functional as F

from mcqa_gradual_das import LearnedSubspace, answer_label_ids, answer_logits, collect_all_layer_states, collect_layer_states, output_iia, output_label_ids, output_targets, run_full_layer_intervention, set_seed

AT_TARGET = "AT_after_AP_intervene"
Y_TARGET = "Y_after_AP_intervene"


def run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, ap_layer, at_layer, output_ids, token_position="last_token", batch_size=8, require_grad=True):
    """AP <- source AP subspace; return output Y and the AT state naturally produced downstream."""
    device = next(model.parameters()).device
    all_y = []
    all_at = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = source_ap[start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        pad = (mask == 0).sum(1)
        position = pad + bank["base_position_by_id"][token_position][start:end].to(device)
        captured = {}

        def ap_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base = hidden[rows, position].float()
            delta = ((donor.float() - base) @ Q_ap) @ Q_ap.T
            new_hidden[rows, position] = (base + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        def at_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured["at"] = hidden[rows, position].float()

        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        at_handle = model.model.layers[at_layer].register_forward_hook(at_hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            y = answer_logits(model, outputs, mask, output_ids)
            at = captured["at"]

            if require_grad:
                all_y.append(y)
                all_at.append(at)
            else:
                all_y.append(y.detach().cpu())
                all_at.append(at.detach().cpu())
        finally:
            ap_handle.remove()
            at_handle.remove()

    return torch.cat(all_y), torch.cat(all_at)


def run_frozen_at_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, token_position="last_token", batch_size=8, require_grad=True):
    """Replay the AT state produced by AP into a fresh clean base run; return Y."""
    device = next(model.parameters()).device
    all_y = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = generated_at[start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        pad = (mask == 0).sum(1)
        position = pad + bank["base_position_by_id"][token_position][start:end].to(device)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base = hidden[rows, position].float()
            delta = ((donor.float() - base) @ Q_at) @ Q_at.T
            new_hidden[rows, position] = (base + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        handle = model.model.layers[at_layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            y = answer_logits(model, outputs, mask, output_ids)

            if require_grad:
                all_y.append(y)
            else:
                all_y.append(y.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_y)


@torch.no_grad()
def evaluate_ap(model, tokenizer, bank, source_ap, Q_ap, ap_layer, Q_at, at_layer, W_at, b_at, token_position, batch_size, output_space):
    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    direct_y, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, ap_layer, at_layer, output_ids, token_position, batch_size, False)
    mediator_y = run_frozen_at_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, token_position, batch_size, False)

    targets = bank["intervention_targets"]
    at_target = targets[AT_TARGET]
    y_target = targets[Y_TARGET]
    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    direct_iia = output_iia(direct_y, y_target, tokenizer, output_space)
    mediator_iia = output_iia(mediator_y, y_target, tokenizer, output_space)
    at_prediction = at_logits.argmax(-1).cpu()
    readout_acc = float((at_prediction == at_target.cpu()).float().mean())

    return direct_iia, mediator_iia, readout_acc


def train_ap(model, tokenizer, at_checkpoint, at_readout_checkpoint, layers=None, subspace_dim=128, ft_size=None, cal_size=None, te_size=None, epochs=50, lr=1e-2, lambda_med=0.0, lambda_readout=0.0, train_batch_size=32, eval_batch_size=32, token_position=None, seed=0, save_path="results/das_ap.pt", banks=None, output_space="full"):
    set_seed(seed)
    device = next(model.parameters()).device
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    at_checkpoint_data = torch.load(at_checkpoint, map_location="cpu", weights_only=False)
    Q_at = at_checkpoint_data["basis"].float().to(device)
    at_layer = int(at_checkpoint_data["layer"])

    readout = torch.load(at_readout_checkpoint, map_location="cpu", weights_only=False)
    W_at = readout["W_AT"].float().to(device)
    b_at = readout["b_AT"].float().to(device)

    if token_position is None:
        token_position = at_checkpoint_data.get("token_position", "last_token")

    train_bank, cal_banks, test_banks = banks
    cal_bank = cal_banks["AP"]
    test_bank = test_banks["AP"]

    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    if layers is None:
        layers = list(range(at_layer))
    else:
        valid_layers = []
        for layer in layers:
            if layer < at_layer:
                valid_layers.append(layer)
        layers = valid_layers

    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)
    layer_scores = []

    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, output_ids, token_position, eval_batch_size)
        target = cal_bank["intervention_targets"][Y_TARGET]
        iia = output_iia(logits, target, tokenizer, output_space)
        layer_scores.append((layer, iia))
        print(f"[AP layer {layer}] iia={iia:.4f}")

    best_layer_score = -1.0
    ap_layer = None

    for layer, score in layer_scores:
        if score > best_layer_score:
            best_layer_score = score
            ap_layer = layer

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_position_by_id"], ap_layer, token_position, eval_batch_size)
    cal_source = cal_source_by_layer[ap_layer]
    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_position_by_id"], ap_layer, token_position, eval_batch_size)

    y_train = train_bank["intervention_targets"][Y_TARGET]
    at_train = train_bank["intervention_targets"][AT_TARGET]

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_score = -1.0
    best_basis = None
    best_epoch = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_bank["base_input_ids"]))
        total_direct_loss = 0.0
        total_mediator_loss = 0.0
        total_readout_loss = 0.0

        for start in range(0, len(permutation), train_batch_size):
            idx = permutation[start:start + train_batch_size]

            mini_positions = {}
            for name, positions in train_bank["base_position_by_id"].items():
                mini_positions[name] = positions[idx]

            mini = {
                "base_input_ids": train_bank["base_input_ids"][idx],
                "base_attention_mask": train_bank["base_attention_mask"][idx],
                "base_position_by_id": mini_positions,
            }

            optimizer.zero_grad()
            Q_ap = alignment.basis()

            direct_y, generated_at = run_ap_intervention_and_capture_at(model, mini, train_source[idx], Q_ap, ap_layer, at_layer, output_ids, token_position, len(idx), True)
            mediator_y = run_frozen_at_mediator(model, mini, generated_at, Q_at, at_layer, output_ids, token_position, len(idx), True)

            y_target = output_targets(y_train[idx], labels, output_space, device)
            at_target = at_train[idx].to(device)
            at_logits = (generated_at @ Q_at) @ W_at.T + b_at

            direct_loss = F.cross_entropy(direct_y, y_target)
            mediator_loss = F.cross_entropy(mediator_y, y_target)
            readout_loss = F.cross_entropy(at_logits, at_target)

            loss = direct_loss
            loss = loss + lambda_med * mediator_loss
            loss = loss + lambda_readout * readout_loss
            loss.backward()
            optimizer.step()

            batch_size = len(idx)
            total_direct_loss += direct_loss.item() * batch_size
            total_mediator_loss += mediator_loss.item() * batch_size
            total_readout_loss += readout_loss.item() * batch_size

        Q_ap = alignment.basis()
        metrics = evaluate_ap(model, tokenizer, cal_bank, cal_source, Q_ap, ap_layer, Q_at, at_layer, W_at, b_at, token_position, eval_batch_size, output_space)
        direct_iia, mediator_iia, readout_acc = metrics

        active_metrics = [direct_iia]
        if lambda_med > 0:
            active_metrics.append(mediator_iia)
        if lambda_readout > 0:
            active_metrics.append(readout_acc)

        score = min(active_metrics)
        num_train = len(permutation)

        direct_average = total_direct_loss / num_train
        mediator_average = total_mediator_loss / num_train
        readout_average = total_readout_loss / num_train
        print(f"[AP epoch {epoch:02d}] direct={direct_average:.4f} med={mediator_average:.4f} readout={readout_average:.4f} | cal={metrics}")

        if score > best_score:
            best_score = score
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch
            best_metrics = metrics

    test_metrics = evaluate_ap(model, tokenizer, test_bank, test_source, best_basis.to(device), ap_layer, Q_at, at_layer, W_at, b_at, token_position, eval_batch_size, output_space)

    result = {
        "variable": "AP",
        "layer": ap_layer,
        "basis": best_basis,
        "token_position": token_position,
        "subspace_dim": subspace_dim,
        "cal": best_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
    }

    save_dir = os.path.dirname(save_path)
    if save_dir == "":
        save_dir = "."

    os.makedirs(save_dir, exist_ok=True)
    torch.save(result, save_path)
    print(f"[AP BEST] layer={ap_layer} epoch={best_epoch} cal={best_metrics} test={test_metrics}")
    return result
