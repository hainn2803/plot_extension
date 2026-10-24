"""Train AT DAS. AT intervention should make Y equal Y_after_AT_intervene."""

import os

import torch
import torch.nn.functional as F

from mcqa_gradual_das import LearnedSubspace, answer_label_ids, collect_all_layer_states, collect_layer_states, output_iia, output_label_ids, output_targets, run_das_intervention, run_full_layer_intervention, set_seed

TARGET = "Y_after_AT_intervene"


@torch.no_grad()
def evaluate_at(model, tokenizer, bank, source_states, Q, layer, token_position, batch_size, output_space):
    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)
    logits = run_das_intervention(model, bank, source_states, Q, layer, output_ids, token_position, batch_size, require_grad=False)
    target = bank["intervention_targets"][TARGET]
    return output_iia(logits, target, tokenizer, output_space)


def train_at(model, tokenizer, layers=None, subspace_dim=128, ft_size=400, cal_size=200, te_size=200, epochs=50, lr=1e-2, train_batch_size=32, eval_batch_size=32, token_position="last_token", seed=0, save_path="results/das_at.pt", banks=None, output_space="full"):
    set_seed(seed)
    device = next(model.parameters()).device
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_bank, cal_banks, test_banks = banks
    cal_bank = cal_banks["AT"]
    test_bank = test_banks["AT"]

    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)
    layer_scores = []

    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, output_ids, token_position, eval_batch_size)
        target = cal_bank["intervention_targets"][TARGET]
        iia = output_iia(logits, target, tokenizer, output_space)
        layer_scores.append((layer, iia))
        print(f"[AT layer {layer}] iia={iia:.4f}")

    best_layer_score = -1.0
    layer = None

    for candidate_layer, score in layer_scores:
        if score > best_layer_score:
            best_layer_score = score
            layer = candidate_layer

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_position_by_id"], layer, token_position, eval_batch_size)
    cal_source = cal_source_by_layer[layer]
    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_position_by_id"], layer, token_position, eval_batch_size)
    target_train = train_bank["intervention_targets"][TARGET]

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_iia = -1.0
    best_basis = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_bank["base_input_ids"]))
        total_loss = 0.0

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
            Q = alignment.basis()

            logits = run_das_intervention(model, mini, train_source[idx], Q, layer, output_ids, token_position, len(idx), require_grad=True)
            target = output_targets(target_train[idx], labels, output_space, device)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        cal_iia = evaluate_at(model, tokenizer, cal_bank, cal_source, alignment.basis(), layer, token_position, eval_batch_size, output_space)
        print(f"[AT epoch {epoch:02d}] loss={total_loss / len(permutation):.4f} cal_iia={cal_iia:.4f}")

        if cal_iia > best_iia:
            best_iia = cal_iia
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch

    test_iia = evaluate_at(model, tokenizer, test_bank, test_source, best_basis.to(device), layer, token_position, eval_batch_size, output_space)

    result = {
        "variable": "AT",
        "layer": layer,
        "basis": best_basis,
        "token_position": token_position,
        "subspace_dim": subspace_dim,
        "cal_iia": best_iia,
        "test_iia": test_iia,
        "best_epoch": best_epoch,
        "config": {"ft_size": ft_size, "cal_size": cal_size, "te_size": te_size},
    }

    save_dir = os.path.dirname(save_path)
    if save_dir == "":
        save_dir = "."

    os.makedirs(save_dir, exist_ok=True)
    torch.save(result, save_path)
    print(f"[AT BEST] layer={layer} epoch={best_epoch} cal={best_iia:.4f} test={test_iia:.4f}")
    return result
