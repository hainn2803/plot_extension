"""Train C DAS and test whether C naturally produces AP, AT, and Y."""

import os

import torch
import torch.nn.functional as F

from mcqa_gradual_das import LearnedSubspace, answer_label_ids, answer_logits, collect_all_layer_states, collect_layer_states, output_iia, output_label_ids, output_targets, run_full_layer_intervention, set_seed


AP_TARGET = "AP_after_C_intervene"
AT_TARGET = "AT_after_C_intervene"
Y_TARGET = "Y_after_C_intervene"


def run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_token_position, list_ap_token_positions, at_token_position, batch_size=8, require_grad=True):
    """Intervene on C, then capture the naturally produced joint AP state and AT state."""
    device = next(model.parameters()).device

    all_y = []
    all_ap = []
    all_at = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor_c = source_c[start:end].to(device)

        num_examples = ids.shape[0]
        rows = torch.arange(num_examples, device=device)
        pad_offset = (mask == 0).sum(dim=1)

        c_position = pad_offset + bank["base_position_by_id"][c_token_position][start:end].to(device)

        ap_positions = []
        for ap_token_position in list_ap_token_positions:
            ap_position = pad_offset + bank["base_position_by_id"][ap_token_position][start:end].to(device)
            ap_positions.append(ap_position)

        ap_positions = torch.stack(ap_positions, dim=1)

        at_position = pad_offset + bank["base_position_by_id"][at_token_position][start:end].to(device)

        captured = {}

        def c_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()

            base_c = hidden[rows, c_position, :].float()
            delta_c = ((donor_c.float() - base_c) @ Q_c) @ Q_c.T
            new_hidden[rows, c_position, :] = (base_c + delta_c).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        def ap_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            ap_state = hidden[rows[:, None], ap_positions, :]
            captured["ap"] = ap_state.float()

        def at_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            at_state = hidden[rows, at_position, :]
            captured["at"] = at_state.float()

        def ap_at_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            ap_state = hidden[rows[:, None], ap_positions, :]
            at_state = hidden[rows, at_position, :]

            captured["ap"] = ap_state.float()
            captured["at"] = at_state.float()

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)

        if ap_layer == at_layer:
            ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_at_hook)
            at_handle = None
        else:
            ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_hook)
            at_handle = model.model.layers[at_layer].register_forward_hook(at_hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)

            y = answer_logits(model, outputs, mask, output_ids)
            ap = captured["ap"]
            at = captured["at"]

            if require_grad:
                all_y.append(y)
                all_ap.append(ap)
                all_at.append(at)
            else:
                all_y.append(y.detach().cpu())
                all_ap.append(ap.detach().cpu())
                all_at.append(at.detach().cpu())

        finally:
            c_handle.remove()
            ap_handle.remove()

            if at_handle is not None:
                at_handle.remove()

    return torch.cat(all_y), torch.cat(all_ap), torch.cat(all_at)


def run_frozen_mediator(model, bank, generated_states, Q, layer, output_ids, list_token_positions, batch_size=8, require_grad=True):
    """Replay generated states through a frozen joint DAS handle."""
    device = next(model.parameters()).device
    all_y = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = generated_states[start:end].to(device)

        num_examples = ids.shape[0]
        num_tokens = len(list_token_positions)

        rows = torch.arange(num_examples, device=device)
        pad_offset = (mask == 0).sum(dim=1)

        token_positions = []

        for token_position in list_token_positions:
            position = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)
            token_positions.append(position)

        token_positions = torch.stack(token_positions, dim=1)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()

            base_states = hidden[rows[:, None], token_positions, :].float()

            base_flat = base_states.reshape(num_examples, -1)
            donor_flat = donor.reshape(num_examples, -1).float()

            delta_flat = ((donor_flat - base_flat) @ Q) @ Q.T
            new_flat = base_flat + delta_flat

            new_states = new_flat.reshape(num_examples, num_tokens, -1)

            new_hidden[rows[:, None], token_positions, :] = new_states.to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
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
def evaluate_c(model, tokenizer, bank, source_c, Q_c, c_layer, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, c_token_position, batch_size, output_space, compute_ap_mediator=True, compute_ap_readout=True, compute_at_mediator=True, compute_at_readout=True):
    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_token_position, list_ap_token_positions, at_token_position, batch_size, False)

    targets = bank["intervention_targets"]

    ap_target = targets[AP_TARGET]
    at_target = targets[AT_TARGET]
    y_target = targets[Y_TARGET]

    direct_iia = output_iia(direct_y, y_target, tokenizer, output_space)

    ap_mediator_iia = 0.0
    if compute_ap_mediator:
        ap_mediator_y = run_frozen_mediator(model, bank, generated_ap, Q_ap, ap_layer, output_ids, list_ap_token_positions, batch_size, False)
        ap_mediator_iia = output_iia(ap_mediator_y, y_target, tokenizer, output_space)

    ap_readout_acc = 0.0
    if compute_ap_readout:
        generated_ap_flat = generated_ap.reshape(generated_ap.shape[0], -1)
        ap_logits = (generated_ap_flat.to(Q_ap.device) @ Q_ap) @ W_ap.T + b_ap
        ap_prediction = ap_logits.argmax(dim=-1).cpu()
        ap_readout_acc = float((ap_prediction == ap_target.cpu()).float().mean())

    at_mediator_iia = 0.0
    if compute_at_mediator:
        list_at_token_positions = [at_token_position]
        at_mediator_y = run_frozen_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, list_at_token_positions, batch_size, False)
        at_mediator_iia = output_iia(at_mediator_y, y_target, tokenizer, output_space)

    at_readout_acc = 0.0
    if compute_at_readout:
        at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
        at_prediction = at_logits.argmax(dim=-1).cpu()
        at_readout_acc = float((at_prediction == at_target.cpu()).float().mean())

    return direct_iia, ap_mediator_iia, ap_readout_acc, at_mediator_iia, at_readout_acc


def train_c(model, tokenizer, ap_checkpoint, ap_readout_checkpoint, at_checkpoint, at_readout_checkpoint, layers=None, subspace_dim=128, ft_size=400, cal_size=200, te_size=200, epochs=50, lr=1e-2, lambda_ap_med=0.0, lambda_ap_readout=0.0, lambda_at_med=0.0, lambda_at_readout=0.0, train_batch_size=32, eval_batch_size=32, token_position="color", seed=0, save_path="results/das_c.pt", banks=None, output_space="full"):
    set_seed(seed)

    device = next(model.parameters()).device
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    ap_checkpoint_data = torch.load(ap_checkpoint, map_location="cpu", weights_only=False)
    Q_ap = ap_checkpoint_data["basis"].float().to(device)
    ap_layer = int(ap_checkpoint_data["layer"])
    list_ap_token_positions = ap_checkpoint_data["token_position"]

    at_checkpoint_data = torch.load(at_checkpoint, map_location="cpu", weights_only=False)
    Q_at = at_checkpoint_data["basis"].float().to(device)
    at_layer = int(at_checkpoint_data["layer"])
    at_token_position = at_checkpoint_data["token_position"]

    ap_readout = torch.load(ap_readout_checkpoint, map_location="cpu", weights_only=False)
    W_ap = ap_readout["W_AP"].float().to(device)
    b_ap = ap_readout["b_AP"].float().to(device)

    at_readout = torch.load(at_readout_checkpoint, map_location="cpu", weights_only=False)
    W_at = at_readout["W_AT"].float().to(device)
    b_at = at_readout["b_AT"].float().to(device)

    if not isinstance(list_ap_token_positions, list):
        raise ValueError(f"AP token_position must be a list, got {type(list_ap_token_positions)}")

    expected_ap_width = len(list_ap_token_positions) * model.config.hidden_size

    if Q_ap.shape[0] != expected_ap_width:
        raise ValueError(f"AP basis width {Q_ap.shape[0]} does not match {len(list_ap_token_positions)} tokens x hidden size {model.config.hidden_size} = {expected_ap_width}")

    if Q_at.shape[0] != model.config.hidden_size:
        raise ValueError(f"AT basis width {Q_at.shape[0]} does not match hidden size {model.config.hidden_size}")

    train_bank, cal_banks, test_banks = banks

    cal_bank = cal_banks["C"]
    test_bank = test_banks["C"]

    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    max_layer = min(ap_layer, at_layer)

    if layers is None:
        layers = list(range(max_layer))
    else:
        valid_layers = []

        for layer in layers:
            if layer < max_layer:
                valid_layers.append(layer)

        layers = valid_layers

    if len(layers) == 0:
        raise ValueError(f"No valid C layers before AP layer {ap_layer} and AT layer {at_layer}")

    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)

    layer_scores = []

    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, output_ids, token_position, eval_batch_size)

        target = cal_bank["intervention_targets"][Y_TARGET]
        iia = output_iia(logits, target, tokenizer, output_space)

        layer_scores.append((layer, iia))

        print(f"[C layer {layer}] iia={iia:.4f}")

    c_layer = None
    best_layer_score = -1.0

    for layer, score in layer_scores:
        if score > best_layer_score:
            best_layer_score = score
            c_layer = layer

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_position_by_id"], c_layer, token_position, eval_batch_size)
    cal_source = cal_source_by_layer[c_layer]
    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_position_by_id"], c_layer, token_position, eval_batch_size)

    train_targets = train_bank["intervention_targets"]

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_score = -1.0
    best_basis = None
    best_epoch = None
    best_metrics = None

    list_at_token_positions = [at_token_position]

    compute_ap_mediator = lambda_ap_med > 0
    compute_ap_readout = lambda_ap_readout > 0
    compute_at_mediator = lambda_at_med > 0
    compute_at_readout = lambda_at_readout > 0

    print(f"[C objectives] direct=1 ap_med={lambda_ap_med} ap_readout={lambda_ap_readout} at_med={lambda_at_med} at_readout={lambda_at_readout}")

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_bank["base_input_ids"]))

        total_direct_loss = 0.0
        total_ap_mediator_loss = 0.0
        total_ap_readout_loss = 0.0
        total_at_mediator_loss = 0.0
        total_at_readout_loss = 0.0

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

            Q_c = alignment.basis()

            direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, mini, train_source[idx], Q_c, c_layer, ap_layer, at_layer, output_ids, token_position, list_ap_token_positions, at_token_position, len(idx), True)

            y_target = output_targets(train_targets[Y_TARGET][idx], labels, output_space, device)

            direct_loss = F.cross_entropy(direct_y, y_target)
            ap_mediator_loss = torch.tensor(0.0, device=device)
            ap_readout_loss = torch.tensor(0.0, device=device)
            at_mediator_loss = torch.tensor(0.0, device=device)
            at_readout_loss = torch.tensor(0.0, device=device)

            if compute_ap_mediator:
                ap_mediator_y = run_frozen_mediator(model, mini, generated_ap, Q_ap, ap_layer, output_ids, list_ap_token_positions, len(idx), True)
                ap_mediator_loss = F.cross_entropy(ap_mediator_y, y_target)

            if compute_ap_readout:
                ap_target = train_targets[AP_TARGET][idx].to(device)
                generated_ap_flat = generated_ap.reshape(generated_ap.shape[0], -1)
                ap_logits = (generated_ap_flat @ Q_ap) @ W_ap.T + b_ap
                ap_readout_loss = F.cross_entropy(ap_logits, ap_target)

            if compute_at_mediator:
                at_mediator_y = run_frozen_mediator(model, mini, generated_at, Q_at, at_layer, output_ids, list_at_token_positions, len(idx), True)
                at_mediator_loss = F.cross_entropy(at_mediator_y, y_target)

            if compute_at_readout:
                at_target = train_targets[AT_TARGET][idx].to(device)
                at_logits = (generated_at @ Q_at) @ W_at.T + b_at
                at_readout_loss = F.cross_entropy(at_logits, at_target)

            loss = direct_loss
            loss = loss + lambda_ap_med * ap_mediator_loss
            loss = loss + lambda_ap_readout * ap_readout_loss
            loss = loss + lambda_at_med * at_mediator_loss
            loss = loss + lambda_at_readout * at_readout_loss

            loss.backward()
            optimizer.step()

            current_batch_size = len(idx)

            total_direct_loss += direct_loss.item() * current_batch_size

            if compute_ap_mediator:
                total_ap_mediator_loss += ap_mediator_loss.item() * current_batch_size

            if compute_ap_readout:
                total_ap_readout_loss += ap_readout_loss.item() * current_batch_size

            if compute_at_mediator:
                total_at_mediator_loss += at_mediator_loss.item() * current_batch_size

            if compute_at_readout:
                total_at_readout_loss += at_readout_loss.item() * current_batch_size

        Q_c = alignment.basis()

        metrics = evaluate_c(model, tokenizer, cal_bank, cal_source, Q_c, c_layer, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, token_position, eval_batch_size, output_space, compute_ap_mediator, compute_ap_readout, compute_at_mediator, compute_at_readout)

        direct_iia, ap_mediator_iia, ap_readout_acc, at_mediator_iia, at_readout_acc = metrics

        active_metrics = [direct_iia]

        if compute_ap_mediator:
            active_metrics.append(ap_mediator_iia)

        if compute_ap_readout:
            active_metrics.append(ap_readout_acc)

        if compute_at_mediator:
            active_metrics.append(at_mediator_iia)

        if compute_at_readout:
            active_metrics.append(at_readout_acc)

        score = min(active_metrics)

        num_train = len(permutation)

        direct_average = total_direct_loss / num_train
        ap_mediator_average = total_ap_mediator_loss / num_train
        ap_readout_average = total_ap_readout_loss / num_train
        at_mediator_average = total_at_mediator_loss / num_train
        at_readout_average = total_at_readout_loss / num_train

        print(f"[C epoch {epoch:02d}] direct={direct_average:.4f} ap_med={ap_mediator_average:.4f} ap_readout={ap_readout_average:.4f} at_med={at_mediator_average:.4f} at_readout={at_readout_average:.4f} | cal={metrics}")

        if score >= best_score:
            best_score = score
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch
            best_metrics = metrics

    full_cal_metrics = evaluate_c(model, tokenizer, cal_bank, cal_source, best_basis.to(device), c_layer, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, token_position, eval_batch_size, output_space)
    test_metrics = evaluate_c(model, tokenizer, test_bank, test_source, best_basis.to(device), c_layer, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, token_position, eval_batch_size, output_space)

    result = {
        "variable": "C",
        "layer": c_layer,
        "basis": best_basis,
        "token_position": token_position,
        "subspace_dim": subspace_dim,
        "cal": full_cal_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
    }

    save_dir = os.path.dirname(save_path)

    if save_dir == "":
        save_dir = "."

    os.makedirs(save_dir, exist_ok=True)

    torch.save(result, save_path)

    print(f"[C BEST] layer={c_layer} epoch={best_epoch} cal={full_cal_metrics} test={test_metrics}")

    return result
