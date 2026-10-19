"""Train AT -> AP -> C, then fit one factual readout for each variable."""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import collect_layer_states, set_seed
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_ap import train_ap
from mcqa_gradual_das_c import train_c
from mcqa_load_randomized_ap_at_c import build_all_banks, build_pair_bank, build_shared_readout_banks, build_ap_readout_datasets, build_at_readout_datasets, build_c_readout_datasets


def parse_layers(x): return None if x == "all" else [int(v) for v in x.split(",")]


def fit_readout(model, checkpoint, save_path, bank_pair, target, token_position, batch_size, seed, num_classes, extra=None):
    """Fit a linear classifier on the frozen DAS coordinates."""
    set_seed(seed); device = next(model.parameters()).device; ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False); Q = ckpt["basis"].float().to(device)

    def features(bank):
        h = collect_layer_states(model, bank["input_ids"], bank["attention_mask"], bank["position_by_id"], ckpt["layer"], token_position, batch_size)
        return (h.to(device) @ Q).detach(), bank["target_ids"].to(device)

    train, val = bank_pair; x, y = features(train); xv, yv = features(val)
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6); x, xv = (x - mean) / std, (xv - mean) / std
    best_acc, best = -1.0, None

    for penalty in (1e-4, 1e-3, 1e-2):
        readout = torch.nn.Linear(Q.shape[1], num_classes).to(device); opt = torch.optim.Adam(readout.parameters(), lr=1e-2)
        for _ in range(200):
            opt.zero_grad(); loss = F.cross_entropy(readout(x), y) + penalty * readout.weight.square().sum(); loss.backward(); opt.step()
            acc = float((readout(xv).argmax(-1) == yv).float().mean())
            if acc > best_acc:
                W = readout.weight.detach() / std.unsqueeze(0); b = readout.bias.detach() - W @ mean
                best_acc = acc; best = {f"W_{target}": W.cpu(), f"b_{target}": b.cpu()}
                if extra: best.update(extra)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True); torch.save(best, save_path); print(f"[{target} readout] val_acc={best_acc:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="randomized_copycolors"); p.add_argument("--result-dir", default="results_joint_randomized")
    p.add_argument("--ft-size", type=int, default=800); p.add_argument("--cal-size", type=int, default=400); p.add_argument("--te-size", type=int, default=400)
    p.add_argument("--at-dim", type=int, default=128); p.add_argument("--ap-dim", type=int, default=128); p.add_argument("--c-dim", type=int, default=128)
    p.add_argument("--at-layers", default="all"); p.add_argument("--ap-layers", default="all"); p.add_argument("--c-layers", default="all")
    p.add_argument("--at-epochs", type=int, default=20); p.add_argument("--ap-epochs", type=int, default=20); p.add_argument("--c-epochs", type=int, default=20); p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--ap-lambda-med", type=float, default=0.0); p.add_argument("--ap-lambda-readout", type=float, default=0.0)
    p.add_argument("--c-lambda-ap-med", type=float, default=0.0); p.add_argument("--c-lambda-ap-readout", type=float, default=0.0); p.add_argument("--c-lambda-at-med", type=float, default=0.0); p.add_argument("--c-lambda-at-readout", type=float, default=0.0)
    p.add_argument("--train-batch-size", type=int, default=32); p.add_argument("--eval-batch-size", type=int, default=32); p.add_argument("--readout-train-size", type=int, default=1300); p.add_argument("--readout-val-size", type=int, default=520)
    p.add_argument("--token-position", default="last_token"); p.add_argument("--c-token-position", default="color"); p.add_argument("--output-space", choices=("full", "az"), default="full"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); set_seed(a.seed)

    result = Path(a.result_dir); result.mkdir(parents=True, exist_ok=True)
    at_path, ap_path, c_path = result / "das_at.pt", result / "das_ap.pt", result / "das_c.pt"
    at_ro, ap_ro, c_ro = result / "at_readout_LogisticRegression.pt", result / "ap_readout_LogisticRegression.pt", result / "c_readout_LogisticRegression.pt"

    model, tokenizer = load_gemma_model()
    das_banks = build_all_banks(model, tokenizer, a.data_path, a.ft_size, a.cal_size, a.te_size, a.eval_batch_size, a.seed)
    shared_readout = build_shared_readout_banks(model, tokenizer, a.data_path, a.readout_train_size, a.readout_val_size, a.eval_batch_size, a.seed)

    print("\n=== TRAIN AT ===")
    train_at(model, tokenizer, parse_layers(a.at_layers), a.at_dim, a.ft_size, a.cal_size, a.te_size, a.at_epochs, a.lr, a.train_batch_size, a.eval_batch_size, a.token_position, a.seed, str(at_path), das_banks["AT"], a.output_space)
    fit_readout(model, at_path, at_ro, build_at_readout_datasets(shared_readout), "AT", a.token_position, a.eval_batch_size, a.seed, 26)

    print("\n=== TRAIN AP ===")
    train_ap(model, tokenizer, str(at_path), str(at_ro), parse_layers(a.ap_layers), a.ap_dim, a.ft_size, a.cal_size, a.te_size, a.ap_epochs, a.lr, a.ap_lambda_med, a.ap_lambda_readout, a.train_batch_size, a.eval_batch_size, a.token_position, a.seed, str(ap_path), das_banks["AP"], a.output_space)
    fit_readout(model, ap_path, ap_ro, build_ap_readout_datasets(shared_readout), "AP", a.token_position, a.eval_batch_size, a.seed, 4)

    print("\n=== TRAIN C ===")
    train_c(model, tokenizer, str(ap_path), str(ap_ro), str(at_path), str(at_ro), parse_layers(a.c_layers), a.c_dim, a.ft_size, a.cal_size, a.te_size, a.c_epochs, a.lr, a.c_lambda_ap_med, a.c_lambda_ap_readout, a.c_lambda_at_med, a.c_lambda_at_readout, a.train_batch_size, a.eval_batch_size, a.c_token_position, a.seed, str(c_path), das_banks["C"], a.output_space)
    colors = shared_readout[0]["color_to_class"]
    fit_readout(model, c_path, c_ro, build_c_readout_datasets(shared_readout), "C", a.c_token_position, a.eval_batch_size, a.seed, len(colors), {"color_to_class": colors})


if __name__ == "__main__": main()
