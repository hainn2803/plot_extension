"""Train AT DAS. AT intervention should make Y equal Y_after_AT_intervene."""

import os, torch
import torch.nn.functional as F
from mcqa_gradual_das import set_seed, answer_label_ids, output_label_ids, output_targets, output_iia, collect_layer_states, collect_all_layer_states, LearnedSubspace, run_full_layer_intervention, run_das_intervention

TARGET = "Y_after_AT_intervene"


@torch.no_grad()
def evaluate_at(model, tokenizer, bank, source_states, Q, layer, token_position, batch_size, output_space):
    labels = answer_label_ids(tokenizer); output_ids = output_label_ids(labels, output_space)
    logits = run_das_intervention(model, bank, source_states, Q, layer, output_ids, token_position, batch_size, require_grad=False)
    return output_iia(logits, bank["intervention_targets"][TARGET], tokenizer, output_space)


def train_at(model, tokenizer, layers=None, subspace_dim=128, ft_size=400, cal_size=200, te_size=200, epochs=50, lr=1e-2, train_batch_size=32, eval_batch_size=32, token_position="last_token", seed=0, save_path="results/das_at.pt", banks=None, output_space="full"):
    set_seed(seed); device = next(model.parameters()).device; model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    train_bank, cal_banks, test_banks = banks; cal_bank = cal_banks["AT"]; test_bank = test_banks["AT"]
    labels = answer_label_ids(tokenizer); output_ids = output_label_ids(labels, output_space)
    layers = list(range(model.config.num_hidden_layers)) if layers is None else layers

    # 1) Pick the layer whose full hidden-state interchange works best.
    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)
    layer_scores = []
    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, output_ids, token_position, eval_batch_size)
        iia = output_iia(logits, cal_bank["intervention_targets"][TARGET], tokenizer, output_space)
        layer_scores.append((layer, iia)); print(f"[AT layer {layer}] iia={iia:.4f}")
    layer = max(layer_scores, key=lambda x: x[1])[0]

    # 2) Freeze model and learn only the AT subspace Q.
    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_position_by_id"], layer, token_position, eval_batch_size)
    cal_source = cal_source_by_layer[layer]
    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_position_by_id"], layer, token_position, eval_batch_size)
    target_train = train_bank["intervention_targets"][TARGET]
    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device); optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)
    best_iia, best_basis, best_epoch = -1.0, None, None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train_bank["base_input_ids"])); total_loss = 0.0
        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]
            mini = {"base_input_ids": train_bank["base_input_ids"][idx], "base_attention_mask": train_bank["base_attention_mask"][idx], "base_position_by_id": {k: v[idx] for k, v in train_bank["base_position_by_id"].items()}}
            optimizer.zero_grad(); Q = alignment.basis()
            logits = run_das_intervention(model, mini, train_source[idx], Q, layer, output_ids, token_position, len(idx), require_grad=True)
            loss = F.cross_entropy(logits, output_targets(target_train[idx], labels, output_space, device)); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(idx)
        cal_iia = evaluate_at(model, tokenizer, cal_bank, cal_source, alignment.basis(), layer, token_position, eval_batch_size, output_space)
        print(f"[AT epoch {epoch:02d}] loss={total_loss / len(perm):.4f} cal_iia={cal_iia:.4f}")
        if cal_iia > best_iia: best_iia, best_basis, best_epoch = cal_iia, alignment.basis().detach().cpu().clone(), epoch

    test_iia = evaluate_at(model, tokenizer, test_bank, test_source, best_basis.to(device), layer, token_position, eval_batch_size, output_space)
    result = {"variable": "AT", "layer": layer, "basis": best_basis, "token_position": token_position, "subspace_dim": subspace_dim, "cal_iia": best_iia, "test_iia": test_iia, "best_epoch": best_epoch, "config": {"ft_size": ft_size, "cal_size": cal_size, "te_size": te_size}}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True); torch.save(result, save_path)
    print(f"[AT BEST] layer={layer} epoch={best_epoch} cal={best_iia:.4f} test={test_iia:.4f}")
    return result
