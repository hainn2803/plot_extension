import argparse, random
from pathlib import Path
import torch
import torch.nn.functional as F

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_c import build_all_banks, color_position, load_generated_dataset, rows_to_pairs, predict_correct, create_bank
from mcqa_data_load_jchang_disjoint import build_readout_datasets
from mcqa_gradual_das import collect_layer_states, set_seed
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_ap import train_ap
from mcqa_gradual_das_c import train_c


def layers(x):
    return None if x == "all" else [int(v) for v in x.split(",") if v.strip()]


def fit_readout(model, checkpoint_path, output_path, args, target, readout_banks, token_position=None, num_classes=None, extra_state=None):
    set_seed(args.seed); ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False); device = next(model.parameters()).device
    Q = ckpt["basis"].to(device).float(); fit, cal = readout_banks
    num_classes = (26 if target == "AT" else 4) if num_classes is None else num_classes
    token_position = args.token_position if token_position is None else token_position

    def data(bank):
        h = collect_layer_states(model, bank["input_ids"], bank["attention_mask"], bank["position_by_id"], ckpt["layer"], token_position, args.eval_batch_size)
        return (h.to(device).float() @ Q).detach(), bank["target_ids"].long().to(device)

    x, y = data(fit); xv, yv = data(cal); mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / std; xv = (xv - mean) / std
    best_acc, best_loss, best = -1.0, float("inf"), None

    for penalty in (1e-4, 1e-3, 1e-2):
        set_seed(args.seed); readout = torch.nn.Linear(Q.shape[1], num_classes).to(device); opt = torch.optim.Adam(readout.parameters(), lr=1e-2)
        for epoch in range(1, 201):
            opt.zero_grad(); logits = readout(x); loss = F.cross_entropy(logits, y) + penalty * readout.weight.square().sum(); loss.backward(); opt.step()
            with torch.no_grad():
                vlogits = readout(xv); vacc = float((vlogits.argmax(-1) == yv).float().mean()); vloss = float(F.cross_entropy(vlogits, yv))
                if vacc > best_acc or (vacc == best_acc and vloss < best_loss):
                    best_acc, best_loss, best_penalty, best_epoch = vacc, vloss, penalty, epoch
                    train_acc = float((readout(x).argmax(-1) == y).float().mean())
                    W = readout.weight.detach() / std.unsqueeze(0); b = readout.bias.detach() - W @ mean
                    best = {f"W_{target}": W.cpu().clone(), f"b_{target}": b.cpu().clone()}
                    if extra_state is not None: best.update(extra_state)
        print(f"[{target} readout] penalty={penalty} done")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True); torch.save(best, output_path)
    print(f"[{target} readout] penalty={best_penalty} epoch={best_epoch} train_acc={train_acc:.4f} balanced_val_acc={best_acc:.4f}; saved to {output_path}")


def get_color_mapping(data_path):
    ds = load_generated_dataset(data_path); colors = set()
    for split in ("train", "validation", "test"):
        for row in ds[split]:
            colors.add(str(row["color"]).strip().casefold())
            colors.add(str(row["color_counterfactual"]["color"]).strip().casefold())
            colors.add(str(row["color_all_counterfactual"]["color"]).strip().casefold())
    colors = sorted(colors); color_to_class = {c: i for i, c in enumerate(colors)}
    print(f"[C classes] {len(colors)} colors: {colors}")
    return color_to_class


@torch.no_grad()
def build_c_readout_datasets(model, tokenizer, data_path, color_to_class, examples_per_class=50, val_examples_per_class=20, batch_size=32, max_attempts_per_class=500, seed=0):
    ds = load_generated_dataset(data_path); device = next(model.parameters()).device; tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    def candidates(row):
        xs = [
            {"prompt": row["prompt"], "answer": row["answer"], "color": row["color"], "question": row["question_key"]},
            {"prompt": row["color_counterfactual"]["prompt"], "answer": row["color_counterfactual"]["answer"], "color": row["color_counterfactual"]["color"], "question": row["question_key"]},
            {"prompt": row["color_all_counterfactual"]["prompt"], "answer": row["color_all_counterfactual"]["answer"], "color": row["color_all_counterfactual"]["color"], "question": row["question_key"]},
        ]
        for x in xs:
            x["color"] = str(x["color"]).strip().casefold(); x["target_id"] = color_to_class[x["color"]]
        return xs

    def build(split, quota, split_seed):
        rng = random.Random(split_seed); accepted = {i: [] for i in range(len(color_to_class))}; seen = set(); ids = list(range(len(ds[split]))); rng.shuffle(ids)
        attempts = 0
        while attempts < max_attempts_per_class and not all(len(v) >= quota for v in accepted.values()):
            attempts += 1; batch = []
            for i in ids:
                for x in candidates(ds[split][i]):
                    if len(accepted[x["target_id"]]) >= quota or x["prompt"] in seen: continue
                    seen.add(x["prompt"]); batch.append(x)
                    if len(batch) >= batch_size: break
                if len(batch) >= batch_size: break
            if not batch: break

            tok = tokenizer([x["prompt"] for x in batch], padding=True, return_tensors="pt").to(device)
            pos = (tok["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
            try: out = model(**tok, position_ids=pos, use_cache=False, logits_to_keep=1)
            except TypeError: out = model(**tok, position_ids=pos, use_cache=False)
            for x, token_id in zip(batch, out.logits[:, -1].argmax(-1).tolist()):
                if tokenizer.decode([int(token_id)]).strip() == x["answer"] and len(accepted[x["target_id"]]) < quota: accepted[x["target_id"]].append(x)
            rng.shuffle(ids)

        missing = {c: quota - len(accepted[i]) for c, i in color_to_class.items() if len(accepted[i]) < quota}
        if missing: raise RuntimeError(f"C readout {split}: insufficient model-correct examples: {missing}")

        xs = [x for i in range(len(color_to_class)) for x in accepted[i]]; rng.shuffle(xs)
        tok = tokenizer([x["prompt"] for x in xs], padding=True, return_tensors="pt")
        bank = {
            "prompts": [x["prompt"] for x in xs], "questions": [x["question"] for x in xs],
            "target_ids": torch.tensor([x["target_id"] for x in xs]), "input_ids": tok["input_ids"].long(), "attention_mask": tok["attention_mask"].long(),
            "position_by_id": {"color": torch.tensor([color_position(x["prompt"], tokenizer) for x in xs]), "last_token": tok["attention_mask"].sum(1).long() - 1},
        }
        print(f"[C readout] {split}: {quota} per class, {len(xs)} prompts"); return bank

    train, val = build("train", examples_per_class, seed), build("validation", val_examples_per_class, seed + 1)
    assert set(train["questions"]).isdisjoint(val["questions"])
    return train, val


def build_c_chain_banks(model, tokenizer, data_path, ft_size, cal_size, te_size, batch_size, seed):
    ds = load_generated_dataset(data_path); sizes = {"train": ft_size, "validation": cal_size, "test": te_size}; banks = {}; cache = {}
    for offset, split in enumerate(("train", "validation", "test")):
        ids = list(range(len(ds[split]))); random.Random(seed + offset).shuffle(ids); pairs = []
        for start in range(0, len(ids), 64):
            cand = rows_to_pairs(ds[split], ids[start:start+64], split)
            cand = [p for p in cand if p["pair_type"] == "C_AP_AT"]
            pairs += predict_correct(model, tokenizer, cand, batch_size, cache)
            if len(pairs) >= sizes[split]: break
        banks[split] = create_bank(pairs, tokenizer, sizes[split], seed + offset)
        print(f"[C chain bank] {split}: {len(banks[split]['pairs'])} C_AP_AT pairs")
    return banks["train"], {"color": banks["validation"]}, {"color": banks["test"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="randomized_copycolors")
    p.add_argument("--at-save-path", default="results_joint_randomized/das_at.pt"); p.add_argument("--ap-save-path", default="results_joint_randomized/das_ap.pt"); p.add_argument("--c-save-path", default="results_joint_randomized/das_c.pt")
    p.add_argument("--at-layers", default="all"); p.add_argument("--ap-layers", default="all"); p.add_argument("--c-layers", default="all")
    p.add_argument("--at-dim", type=int, default=128); p.add_argument("--ap-dim", type=int, default=128); p.add_argument("--c-dim", type=int, default=128)
    p.add_argument("--ft-size", type=int, default=800); p.add_argument("--cal-size", type=int, default=400); p.add_argument("--te-size", type=int, default=400)
    p.add_argument("--at-epochs", type=int, default=20); p.add_argument("--ap-epochs", type=int, default=20); p.add_argument("--c-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lambda-med", "--ap-lambda-med", dest="ap_lambda_med", type=float, default=0.0)
    p.add_argument("--lambda-readout", "--ap-lambda-readout", dest="ap_lambda_readout", type=float, default=0.0)
    p.add_argument("--c-lambda-ap-med", type=float, default=0.0)
    p.add_argument("--c-lambda-ap-readout", type=float, default=0.0)
    p.add_argument("--c-lambda-at-med", type=float, default=0.0)
    p.add_argument("--c-lambda-at-readout", type=float, default=0.0)
    p.add_argument("--train-batch-size", type=int, default=32); p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--token-position", default="last_token"); p.add_argument("--c-token-position", default="color")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--output-space", choices=("full", "az"), default="full")
    p.add_argument("--at-readout-per-class", type=int, default=50); p.add_argument("--ap-readout-per-class", type=int, default=50)
    p.add_argument("--c-readout-per-class", type=int, default=50); p.add_argument("--c-readout-val-per-class", type=int, default=20); p.add_argument("--readout-max-attempts", type=int, default=500)
    a = p.parse_args()

    lambdas = [a.ap_lambda_med, a.ap_lambda_readout, a.c_lambda_ap_med, a.c_lambda_ap_readout, a.c_lambda_at_med, a.c_lambda_at_readout]
    if any(x < 0 for x in lambdas): p.error("all lambda values must be nonnegative")

    model, tokenizer = load_gemma_model()
    all_banks = build_all_banks(model, tokenizer, a.data_path, a.ft_size, a.cal_size, a.te_size, a.eval_batch_size, a.seed)

    print("\n" + "=" * 80 + "\nTRAIN AT\n" + "=" * 80)
    train_at(model, tokenizer, layers=layers(a.at_layers), subspace_dim=a.at_dim, ft_size=a.ft_size, cal_size=a.cal_size, te_size=a.te_size, epochs=a.at_epochs, lr=a.lr, train_batch_size=a.train_batch_size, eval_batch_size=a.eval_batch_size, token_position=a.token_position, seed=a.seed, save_path=a.at_save_path, banks=all_banks["AT"], output_space=a.output_space)

    at_readout_path = str(Path(a.at_save_path).with_name("at_readout_LogisticRegression.pt"))
    at_readout_banks = build_readout_datasets(model, tokenizer, target="AT", examples_per_class=a.at_readout_per_class, batch_size=a.eval_batch_size, max_attempts_per_class=a.readout_max_attempts, seed=a.seed)
    fit_readout(model, a.at_save_path, at_readout_path, a, "AT", at_readout_banks)

    print("\n" + "=" * 80 + "\nTRAIN AP\n" + "=" * 80)
    train_ap(model, tokenizer, at_checkpoint=a.at_save_path, at_readout_checkpoint=at_readout_path, layers=layers(a.ap_layers), subspace_dim=a.ap_dim, ft_size=a.ft_size, cal_size=a.cal_size, te_size=a.te_size, epochs=a.ap_epochs, lr=a.lr, lambda_med=a.ap_lambda_med, lambda_readout=a.ap_lambda_readout, train_batch_size=a.train_batch_size, eval_batch_size=a.eval_batch_size, token_position=a.token_position, seed=a.seed, save_path=a.ap_save_path, banks=all_banks["AP"], output_space=a.output_space)

    ap_readout_path = str(Path(a.ap_save_path).with_name("ap_readout_LogisticRegression.pt"))
    ap_readout_banks = build_readout_datasets(model, tokenizer, target="AP", examples_per_class=a.ap_readout_per_class, batch_size=a.eval_batch_size, max_attempts_per_class=a.readout_max_attempts, seed=a.seed)
    fit_readout(model, a.ap_save_path, ap_readout_path, a, "AP", ap_readout_banks)

    print("\n" + "=" * 80 + "\nTRAIN C\n" + "=" * 80)
    c_banks = build_c_chain_banks(model, tokenizer, a.data_path, a.ft_size, a.cal_size, a.te_size, a.eval_batch_size, a.seed)
    train_c(model, tokenizer, ap_checkpoint=a.ap_save_path, ap_readout_checkpoint=ap_readout_path, at_checkpoint=a.at_save_path, at_readout_checkpoint=at_readout_path, layers=layers(a.c_layers), subspace_dim=a.c_dim, ft_size=a.ft_size, cal_size=a.cal_size, te_size=a.te_size, epochs=a.c_epochs, lr=a.lr, lambda_ap_med=a.c_lambda_ap_med, lambda_ap_readout=a.c_lambda_ap_readout, lambda_at_med=a.c_lambda_at_med, lambda_at_readout=a.c_lambda_at_readout, train_batch_size=a.train_batch_size, eval_batch_size=a.eval_batch_size, token_position=a.c_token_position, seed=a.seed, save_path=a.c_save_path, banks=c_banks, output_space=a.output_space)

    color_to_class = get_color_mapping(a.data_path)
    c_readout_banks = build_c_readout_datasets(model, tokenizer, a.data_path, color_to_class, a.c_readout_per_class, a.c_readout_val_per_class, a.eval_batch_size, a.readout_max_attempts, a.seed)
    c_readout_path = str(Path(a.c_save_path).with_name("c_readout_LogisticRegression.pt"))
    fit_readout(model, a.c_save_path, c_readout_path, a, "C", c_readout_banks, token_position=a.c_token_position, num_classes=len(color_to_class), extra_state={"color_to_class": color_to_class})

    print("\n" + "=" * 80 + "\nDONE\n" + "=" * 80)


if __name__ == "__main__": main()
