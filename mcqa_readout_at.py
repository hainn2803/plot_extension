import torch

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score

from mcqa_neural_net import load_gemma_model
from mcqa_data_load_all import build_mcqa_banks
from mcqa_gradual_das_at import collect_layer_states


def train_probe(probe_model, X_train, y_train, X_test, y_test):
    probe_model.fit(X_train, y_train)

    if isinstance(probe_model, LinearRegression):
        train_pred = probe_model.predict(X_train)
        test_pred = probe_model.predict(X_test)

        print("Train R2:", r2_score(y_train, train_pred))
        print("Test R2:", r2_score(y_test, test_pred))
    else:
        print("Train accuracy:", probe_model.score(X_train, y_train))
        print("Test accuracy:", probe_model.score(X_test, y_test))

    return probe_model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_gemma_model()

    at_checkpoint = "results/das_at.pt"
    at_result = torch.load(at_checkpoint, map_location="cpu")
    at_layer = int(at_result["layer"])
    Q_at = at_result["basis"].float()

    config = at_result["config"]
    fit_bank, cal_banks, te_banks = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=config["ft_size"], cal_size=config["cal_size"], te_size=config["te_size"], dataset_size=None, split="train", device=device, batch_size=config["eval_batch_size"], seed=config["seed"])
    te_bank = te_banks["answer_token"]

    X_train = collect_layer_states(model, fit_bank["source_input_ids"], fit_bank["source_attention_mask"], fit_bank["source_position_by_id"], at_layer, token_position=at_result["token_position"], batch_size=config["eval_batch_size"])
    X_test = collect_layer_states(model, te_bank["source_input_ids"], te_bank["source_attention_mask"], te_bank["source_position_by_id"], at_layer, token_position=at_result["token_position"], batch_size=config["eval_batch_size"])

    X_train = (X_train @ Q_at).numpy()
    X_test = (X_test @ Q_at).numpy()

    y_train = fit_bank["counterfactual_label_ids"]["answer_token"].numpy()
    y_test = te_bank["counterfactual_label_ids"]["answer_token"].numpy()

    probe_model = LogisticRegression(max_iter=10000)
    trained_probe = train_probe(probe_model, X_train, y_train, X_test, y_test)

    W_at = torch.from_numpy(trained_probe.coef_).float()
    b_at = torch.from_numpy(trained_probe.intercept_).float()

    torch.save({
        "W_AT": W_at,
        "b_AT": b_at,
        "classes": torch.from_numpy(trained_probe.classes_)
    }, "results/at_readout_LogisticRegression.pt")