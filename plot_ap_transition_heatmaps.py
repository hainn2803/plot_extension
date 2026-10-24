import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRANSITIONS = [(i, j) for i in range(4) for j in range(4) if i != j]


def load_transition(input_dir, base_ap, source_ap):
    path = Path(input_dir) / f"ap_{base_ap}_to_{source_ap}_full_layer_token_sweep.csv"
    df = pd.read_csv(path)
    matrix = df.pivot(index="layer", columns="token_index", values="iia").sort_index().sort_index(axis=1)
    token_rows = df.sort_values(["token_index", "layer"]).drop_duplicates("token_index")
    token_labels = {int(row.token_index): str(row.base_token) for _, row in token_rows.iterrows()}
    return matrix, token_labels


def short_token(token):
    token = token.replace("\n", "\\n")
    if len(token) > 10:
        token = token[:9] + "…"
    return token


def draw_heatmap(ax, matrix, token_labels, title, show_y=True, show_x=True):
    image = ax.imshow(matrix.values, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Token index from end" if show_x else "")
    ax.set_ylabel("Layer" if show_y else "")

    x = np.arange(len(matrix.columns))
    y = np.arange(len(matrix.index))

    if show_x:
        labels = [f"{int(t)}\n{short_token(token_labels[int(t)])}" for t in matrix.columns]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])

    if show_y:
        ax.set_yticks(y)
        ax.set_yticklabels(matrix.index)
    else:
        ax.set_yticks([])

    return image


def save_individual(input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for base_ap, source_ap in TRANSITIONS:
        matrix, token_labels = load_transition(input_dir, base_ap, source_ap)
        fig, ax = plt.subplots(figsize=(14, 7))
        image = draw_heatmap(ax, matrix, token_labels, f"AP {base_ap}→{source_ap}: full-vector patch IIA")
        fig.colorbar(image, ax=ax, label="Counterfactual accuracy / IIA")
        fig.tight_layout()
        path = output_dir / f"ap_{base_ap}_to_{source_ap}_heatmap.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


def save_combined(input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True)
    image = None

    for idx, (base_ap, source_ap) in enumerate(TRANSITIONS):
        row, col = divmod(idx, 4)
        matrix, token_labels = load_transition(input_dir, base_ap, source_ap)
        image = draw_heatmap(axes[row, col], matrix, token_labels, f"{base_ap}→{source_ap}", show_y=(col == 0), show_x=(row == 2))

    cbar = fig.colorbar(image, ax=axes, shrink=0.9)
    cbar.set_label("Counterfactual accuracy / IIA")
    fig.suptitle("Full-vector AP localization by ordered transition", fontsize=18)
    path = output_dir / "ap_all_transitions_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="results_ap_transition_token_sweep")
    p.add_argument("--output-dir", default="results_ap_transition_token_sweep/heatmaps")
    a = p.parse_args()

    save_individual(a.input_dir, a.output_dir)
    save_combined(a.input_dir, a.output_dir)


if __name__ == "__main__":
    main()
