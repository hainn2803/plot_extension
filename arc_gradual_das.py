import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

from arc_data_load_all import build_arc_banks
from mcqa_gradual_das import (
    LearnedSubspace,
    answer_label_ids,
    collect_all_layer_states,
    collect_layer_states,
    run_das_intervention,
    run_full_layer_intervention,
    set_seed,
)
from mcqa_gradual_das_ap import (
    evaluate_ap,
    run_ap_intervention_and_capture_at,
    run_frozen_at_mediator,
)


MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKEN_POSITION = "last_token"


def load_model():
    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=token, torch_dtype=dtype
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


@torch.no_grad()
def answer_iia(model, bank, source_states, Q, layer, labels, batch_size):
    logits = run_das_intervention(
        model, bank, source_states, Q, layer, labels,
        TOKEN_POSITION, batch_size, require_grad=False,
    )
    target = bank["counterfactual_label_ids"]["answer_token"]
    return float((logits.argmax(dim=-1) == target).float().mean())


def make_answer_checkpoint(
    model, labels, layer, Q, epoch, cal_iia, test_iia, config
):
    return {
        "variable": "answer_token",
        "method": "DAS learned orthonormal subspace",
        "layer": layer,
        "token_position": TOKEN_POSITION,
        "hidden_size": int(model.config.hidden_size),
        "subspace_dim": int(Q.shape[1]),
        "basis": Q.detach().cpu(),
        "cal_iia": float(cal_iia),
        "test_iia": test_iia,
        "best_epoch": int(epoch),
        "answer_label_ids": labels,
        "config": config,
    }


def train_answer(
    model, tokenizer, banks, k, epochs, lr, batch_size,
    save_dir, save_every, config,
):
    device = next(model.parameters()).device
    labels = answer_label_ids(tokenizer)
    fit, cal, test = banks["fit"], banks["cal"], banks["test"]
    layers = list(range(model.config.num_hidden_layers))

    cal_source_all = collect_all_layer_states(
        model, cal["source_input_ids"], cal["source_attention_mask"],
        cal["source_position_by_id"], layers, TOKEN_POSITION, batch_size,
    )

    layer_scores = []
    for layer in layers:
        logits = run_full_layer_intervention(
            model, cal, cal_source_all[layer], layer,
            labels, TOKEN_POSITION, batch_size,
        )
        target = cal["counterfactual_label_ids"]["answer_token"]
        score = float((logits.argmax(dim=-1) == target).float().mean())
        layer_scores.append(score)
        print(f"[OAnswer layer] layer={layer} iia={score:.4f}")

    best_iia = max(layer_scores)
    layer = max(i for i, score in enumerate(layer_scores) if score == best_iia)

    print(f"[OAnswer layer selected] layer={layer} iia={best_iia:.4f}")
    # print(f"[OAnswer layer selected] layer={layer} iia={layer_scores[layer]:.4f}")

    fit_source = collect_layer_states(
        model, fit["source_input_ids"], fit["source_attention_mask"],
        fit["source_position_by_id"], layer, TOKEN_POSITION, batch_size,
    )
    test_source = collect_layer_states(
        model, test["source_input_ids"], test["source_attention_mask"],
        test["source_position_by_id"], layer, TOKEN_POSITION, batch_size,
    )

    alignment = LearnedSubspace(model.config.hidden_size, k).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_cal = -1.0
    best_Q = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(fit["base_input_ids"]))
        total_loss = 0.0
        total_count = 0

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            mini = {
                "base_input_ids": fit["base_input_ids"][idx],
                "base_attention_mask": fit["base_attention_mask"][idx],
                "base_position_by_id": {
                    TOKEN_POSITION: fit["base_position_by_id"][TOKEN_POSITION][idx]
                },
            }

            Q = alignment.basis()
            logits = run_das_intervention(
                model, mini, fit_source[idx], Q, layer, labels,
                TOKEN_POSITION, len(idx), require_grad=True,
            )
            target = fit["counterfactual_label_ids"]["answer_token"][idx].to(device)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_count += len(idx)

        with torch.no_grad():
            Q = alignment.basis()
            cal_iia = answer_iia(
                model, cal, cal_source_all[layer], Q,
                layer, labels, batch_size,
            )

        print(
            f"[OAnswer epoch {epoch:02d}] "
            f"loss={total_loss / total_count:.4f} "
            f"cal_iia={cal_iia:.4f}"
        )

        if epoch % save_every == 0 or epoch == epochs:
            checkpoint = make_answer_checkpoint(
                model, labels, layer, Q, epoch, cal_iia, None, config
            )
            torch.save(
                checkpoint,
                os.path.join(save_dir, f"oanswer_epoch_{epoch:03d}.pt"),
            )

        if cal_iia > best_cal:
            best_cal = cal_iia
            best_Q = Q.detach().cpu().clone()
            best_epoch = epoch

    test_iia = answer_iia(
        model, test, test_source, best_Q.to(device),
        layer, labels, batch_size,
    )

    result = make_answer_checkpoint(
        model, labels, layer, best_Q,
        best_epoch, best_cal, test_iia, config,
    )
    torch.save(result, os.path.join(save_dir, "oanswer.pt"))

    print(
        f"[OAnswer] layer={layer} epoch={best_epoch} "
        f"cal={best_cal:.3f} test={test_iia:.3f}"
    )

    return layer, best_Q


def train_readout(model, bank, layer, Q, batch_size):
    device = next(model.parameters()).device
    Q = Q.to(device)

    base = collect_layer_states(
        model, bank["base_input_ids"], bank["base_attention_mask"],
        bank["base_position_by_id"], layer, TOKEN_POSITION, batch_size,
    )
    source = collect_layer_states(
        model, bank["source_input_ids"], bank["source_attention_mask"],
        bank["source_position_by_id"], layer, TOKEN_POSITION, batch_size,
    )

    base_x = (base.to(device) @ Q).float().cpu().numpy()
    source_x = (source.to(device) @ Q).float().cpu().numpy()

    base_y = bank["base_answer_label_ids"].cpu().numpy()
    source_y = bank["source_answer_label_ids"].cpu().numpy()

    x = torch.cat([base, source]).to(device) @ Q
    x = x.float().cpu().numpy()
    y = torch.cat([
        bank["base_answer_label_ids"],
        bank["source_answer_label_ids"],
    ]).cpu().numpy()

    readout = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=0,
    )
    readout.fit(x, y)

    train_acc = float(readout.score(x, y))
    base_acc = float(readout.score(base_x, base_y))
    source_acc = float(readout.score(source_x, source_y))

    print(
        f"[OAnswer readout] "
        f"train_acc={train_acc:.4f} "
        f"base_acc={base_acc:.4f} "
        f"source_acc={source_acc:.4f}"
    )

    W = torch.zeros(26, Q.shape[1], dtype=torch.float32)
    b = torch.full((26,), -1e9, dtype=torch.float32)

    classes = torch.tensor(readout.classes_, dtype=torch.long)
    W[classes] = torch.from_numpy(readout.coef_).float()
    b[classes] = torch.from_numpy(readout.intercept_).float()

    return W, b, train_acc


def make_order_checkpoint(
    layer, answer_layer, Q, epoch,
    direct_cal, med_cal, readout_cal, score,
    lambda_med, lambda_readout, config,
    answer_checkpoint, readout_checkpoint, test=None,
):
    return {
        "variable": "answer_pointer",
        "method": "Gradual DAS",
        "layer": layer,
        "ap_layer": layer,
        "at_layer": answer_layer,
        "basis": Q.detach().cpu(),
        "ap_basis": Q.detach().cpu(),
        "token_position": TOKEN_POSITION,
        "lambda_med": lambda_med,
        "lambda_readout": lambda_readout,
        "cal": {
            "selected_epoch": int(epoch),
            "best_epoch": int(epoch),
            "direct_iia": float(direct_cal),
            "mediator_iia": float(med_cal),
            "readout_accuracy": float(readout_cal),
            "score": float(score),
        },
        "test": test,
        "at_checkpoint": answer_checkpoint,
        "at_readout_checkpoint": readout_checkpoint,
        "config": config,
    }


def train_order(
    model, tokenizer, banks, answer_layer, Q_answer, W, b,
    k, epochs, lr, lambda_med, lambda_readout, batch_size,
    save_dir, save_every, config, answer_checkpoint, readout_checkpoint,
):
    device = next(model.parameters()).device
    labels = answer_label_ids(tokenizer)
    fit, cal, test = banks["fit"], banks["cal"], banks["test"]

    Q_answer = Q_answer.to(device)
    W, b = W.to(device), b.to(device)

    # layers = list(range(answer_layer))
    layers = list(range(model.config.num_hidden_layers))
    cal_source_all = collect_all_layer_states(
        model, cal["source_input_ids"], cal["source_attention_mask"],
        cal["source_position_by_id"], layers, TOKEN_POSITION, batch_size,
    )

    layer_scores = []
    for layer in layers:
        logits = run_full_layer_intervention(
            model, cal, cal_source_all[layer], layer,
            labels, TOKEN_POSITION, batch_size,
        )
        target = cal["counterfactual_label_ids"]["answer_pointer"]
        score = float((logits[:, :4].argmax(dim=-1) == target).float().mean())
        layer_scores.append(score)
        print(f"[XOrder layer] layer={layer} iia={score:.4f}")

    best_layer_idx = max(range(len(layer_scores)), key=layer_scores.__getitem__)
    layer = layers[best_layer_idx]
    print(f"[XOrder layer selected] layer={layer} iia={layer_scores[best_layer_idx]:.4f}")

    fit_source = collect_layer_states(
        model, fit["source_input_ids"], fit["source_attention_mask"],
        fit["source_position_by_id"], layer, TOKEN_POSITION, batch_size,
    )
    test_source = collect_layer_states(
        model, test["source_input_ids"], test["source_attention_mask"],
        test["source_position_by_id"], layer, TOKEN_POSITION, batch_size,
    )

    alignment = LearnedSubspace(model.config.hidden_size, k).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    best_score = -1.0
    best_Q = None
    best_cal = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(fit["base_input_ids"]))

        total_direct_loss = 0.0
        total_med_loss = 0.0
        total_readout_loss = 0.0
        total_count = 0

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            mini = {
                "base_input_ids": fit["base_input_ids"][idx],
                "base_attention_mask": fit["base_attention_mask"][idx],
                "base_position_by_id": {
                    TOKEN_POSITION: fit["base_position_by_id"][TOKEN_POSITION][idx]
                },
            }

            Q_order = alignment.basis()

            direct_logits, generated_answer = run_ap_intervention_and_capture_at(
                model, mini, fit_source[idx], Q_order,
                layer, answer_layer, labels,
                TOKEN_POSITION, len(idx), require_grad=True,
            )
            mediator_logits = run_frozen_at_mediator(
                model, mini, generated_answer, Q_answer,
                answer_layer, labels,
                TOKEN_POSITION, len(idx), require_grad=True,
            )

            pointer_target = fit["counterfactual_label_ids"]["answer_pointer"][idx].to(device)
            answer_target = fit["counterfactual_label_ids"]["answer_token"][idx].to(device)

            # direct_loss = F.cross_entropy(direct_logits[:, :4], pointer_target)
            # mediator_loss = F.cross_entropy(mediator_logits[:, :4], pointer_target)
            direct_loss = F.cross_entropy(direct_logits, pointer_target)
            mediator_loss = F.cross_entropy(mediator_logits, pointer_target)
            readout_loss = F.cross_entropy(
                (generated_answer @ Q_answer) @ W.T + b,
                answer_target,
            )

            loss = (
                direct_loss
                + lambda_med * mediator_loss
                + lambda_readout * readout_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(idx)
            total_direct_loss += direct_loss.item() * n
            total_med_loss += mediator_loss.item() * n
            total_readout_loss += readout_loss.item() * n
            total_count += n

        with torch.no_grad():
            Q_order = alignment.basis()
            direct_cal, med_cal, readout_cal = evaluate_ap(
                model, cal, cal_source_all[layer],
                Q_order, layer, answer_layer,
                Q_answer, W, b, labels,
                TOKEN_POSITION, batch_size,
            )

        score = direct_cal
        if lambda_med > 0:
            score = min(score, med_cal)
        if lambda_readout > 0:
            score = min(score, readout_cal)

        print(
            f"[XOrder epoch {epoch:02d}] "
            f"direct_loss={total_direct_loss / total_count:.4f} "
            f"med_loss={total_med_loss / total_count:.4f} "
            f"readout_loss={total_readout_loss / total_count:.4f} "
            f"direct_cal={direct_cal:.4f} "
            f"mediator_cal={med_cal:.4f} "
            f"readout_cal={readout_cal:.4f} "
            f"score={score:.4f}"
        )

        if epoch % save_every == 0 or epoch == epochs:
            checkpoint = make_order_checkpoint(
                layer, answer_layer, Q_order, epoch,
                direct_cal, med_cal, readout_cal, score,
                lambda_med, lambda_readout, config,
                answer_checkpoint, readout_checkpoint,
            )
            torch.save(
                checkpoint,
                os.path.join(save_dir, f"xorder_epoch_{epoch:03d}.pt"),
            )

        if score > best_score:
            best_score = score
            best_Q = Q_order.detach().cpu().clone()
            best_cal = (direct_cal, med_cal, readout_cal)
            best_epoch = epoch

    direct_test, med_test, readout_test = evaluate_ap(
        model, test, test_source, best_Q.to(device),
        layer, answer_layer, Q_answer, W, b,
        labels, TOKEN_POSITION, batch_size,
    )

    test = {
        "direct_iia": float(direct_test),
        "mediator_iia": float(med_test),
        "readout_accuracy": float(readout_test),
    }

    result = make_order_checkpoint(
        layer, answer_layer, best_Q, best_epoch,
        best_cal[0], best_cal[1], best_cal[2], best_score,
        lambda_med, lambda_readout, config,
        answer_checkpoint, readout_checkpoint, test=test,
    )
    torch.save(result, os.path.join(save_dir, "xorder.pt"))

    print(
        f"[XOrder TEST] direct={direct_test:.3f} "
        f"med={med_test:.3f} readout={readout_test:.3f}"
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fit-size", type=int, default=400)
    parser.add_argument("--cal-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambda-med", type=float, default=0.5)
    parser.add_argument("--lambda-readout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--output-dir", default="results_arc")
    parser.add_argument("--oanswer-checkpoint", default="results_arc/gradual_das_ap_lambda_med_0.0_lambda_readout_1.0/oanswer.pt")

    args = parser.parse_args()

    set_seed(args.seed)
    config_lambd_name = f"gradual_das_ap_lambda_med_{args.lambda_med}_lambda_readout_{args.lambda_readout}"
    args.output_dir = os.path.join(args.output_dir, config_lambd_name)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model()

    banks = build_arc_banks(
        model, tokenizer,
        args.fit_size, args.cal_size, args.test_size,
        args.batch_size, args.seed,
    )

    config = {
        "ft_size": args.fit_size,
        "cal_size": args.cal_size,
        "te_size": args.test_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "seed": args.seed,
        "save_every": args.save_every,
    }

    readout_checkpoint = os.path.join(args.output_dir, "oanswer_readout.pt")

    if args.oanswer_checkpoint is not None:
        answer_checkpoint = args.oanswer_checkpoint
        answer_result = torch.load(answer_checkpoint, map_location="cpu")
        answer_layer = int(answer_result["layer"])
        Q_answer = answer_result["basis"]
        print(f"[OAnswer] loaded checkpoint: {answer_checkpoint}")
        print(f"[OAnswer] layer={answer_layer} k={Q_answer.shape[1]}")
    else:
        answer_checkpoint = os.path.join(args.output_dir, "oanswer.pt")
        answer_layer, Q_answer = train_answer(
            model, tokenizer, banks["answer_token"],
            args.k, args.epochs, args.lr, args.batch_size,
            args.output_dir, args.save_every, config,
        )

    W, b, readout_acc = train_readout(
        model, banks["answer_token"]["fit"],
        answer_layer, Q_answer, args.batch_size,
    )

    torch.save({
        "variable": "answer_token",
        "method": "LogisticRegression",
        "layer": answer_layer,
        "token_position": TOKEN_POSITION,
        "W_AT": W.cpu(),
        "b_AT": b.cpu(),
        "train_accuracy": readout_acc,
        "at_checkpoint": answer_checkpoint,
    }, readout_checkpoint)

    train_order(
        model, tokenizer, banks["answer_pointer"],
        answer_layer, Q_answer, W, b,
        args.k, args.epochs, args.lr,
        args.lambda_med, args.lambda_readout, args.batch_size,
        args.output_dir, args.save_every, config,
        answer_checkpoint, readout_checkpoint,
    )


if __name__ == "__main__":
    main()
