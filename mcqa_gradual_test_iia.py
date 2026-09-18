import torch

from mcqa_neural_net import load_gemma_model
from mcqa_data_load_all import build_mcqa_banks
from mcqa_gradual_das_at import answer_label_ids, collect_layer_states, run_das_intervention


@torch.no_grad()
def evaluate_iia(
    model,
    bank,
    source_states,
    Q,
    layer,
    label_ids,
    token_position="last_token",
    batch_size=16,
):
    logits = run_das_intervention(
        model=model,
        bank=bank,
        source_states=source_states,
        Q=Q,
        layer=layer,
        label_ids=label_ids,
        token_position=token_position,
        batch_size=batch_size,
        require_grad=False,
    )
    pred = logits.argmax(dim=-1).cpu()
    target = bank["counterfactual_label_ids"]["answer_pointer"].cpu()
    return float((pred == target).float().mean().item())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_gemma_model()

    ap_checkpoint = "results/standard_das_ap.pt"
    ap_result = torch.load(ap_checkpoint, map_location="cpu")

    ap_layer = int(ap_result["ap_layer"])
    config = ap_result["config"]
    labels = answer_label_ids(tokenizer)

    fit_bank, cal_banks, te_banks = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=config["ft_size"], cal_size=config["cal_size"], te_size=config["te_size"], dataset_size=None, split="train", device=device, batch_size=config["eval_batch_size"], seed=config["seed"])
    cal_bank = cal_banks["answer_pointer"]
    te_bank = te_banks["answer_pointer"]

    cal_source = collect_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], ap_layer, token_position=ap_result["token_position"], batch_size=config["eval_batch_size"])
    te_source = collect_layer_states(model, te_bank["source_input_ids"], te_bank["source_attention_mask"], te_bank["source_position_by_id"], ap_layer, token_position=ap_result["token_position"], batch_size=config["eval_batch_size"])

    Q_learned = ap_result["ap_basis"].float().to(device)

    torch.manual_seed(0)
    Q_random = torch.linalg.qr(torch.randn_like(ap_result["ap_basis"].float()), mode="reduced").Q.to(device)

    learned_cal_iia = evaluate_iia(model, cal_bank, cal_source, Q_learned, ap_layer, labels, ap_result["token_position"], config["eval_batch_size"])
    learned_test_iia = evaluate_iia(model, te_bank, te_source, Q_learned, ap_layer, labels, ap_result["token_position"], config["eval_batch_size"])

    random_cal_iia = evaluate_iia(model, cal_bank, cal_source, Q_random, ap_layer, labels, ap_result["token_position"], config["eval_batch_size"])
    random_test_iia = evaluate_iia(model, te_bank, te_source, Q_random, ap_layer, labels, ap_result["token_position"], config["eval_batch_size"])

    print("Learned Q CAL IIA:", learned_cal_iia)
    print("Learned Q TEST IIA:", learned_test_iia)
    print("Random Q CAL IIA:", random_cal_iia)
    print("Random Q TEST IIA:", random_test_iia)