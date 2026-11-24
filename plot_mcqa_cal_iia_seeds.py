#!/usr/bin/env python3
"""Mean/std CAL IIA over MCQA seed summaries; vertical + combined heatmaps.

Example:
  python plot_mcqa_cal_iia_seeds.py --root . --seeds 0 1 2 3 4

Reads:  results_standard_all_layers_controlled_seed{seed}_no_distractor/summary.csv
Uses:   cal_iia (never test_iia)
Writes: cal_iia_heatmaps/cal_iia_mean_std.csv, 3 vertical heatmaps, and combined heatmap
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIABLES = ("AT", "AP", "C")
SEED_PATTERN = re.compile(r"(?:^|_)seed(\d+)(?:_|$)")


def find_summaries(root, folder_pattern, requested_seeds):
    summaries = {}
    for directory in sorted(root.glob(folder_pattern)):
        if not directory.is_dir():
            continue
        match = SEED_PATTERN.search(directory.name)
        if match is None:
            continue
        seed = int(match.group(1))
        if requested_seeds is not None and seed not in requested_seeds:
            continue
        summary = directory / "summary.csv"
        if not summary.is_file():
            print(f"[skip] Missing {summary}")
            continue
        if seed in summaries:
            raise ValueError(f"More than one summary.csv for seed {seed}")
        summaries[seed] = summary

    if requested_seeds is not None:
        missing = sorted(set(requested_seeds) - set(summaries))
        if missing:
            raise FileNotFoundError(f"Missing summary.csv for seeds: {missing}")
    if not summaries:
        raise FileNotFoundError(f"No summary.csv matched {root / folder_pattern}")
    return summaries


def load_cal_iia(summaries):
    # (variable, layer) -> {seed: CAL IIA}
    values = defaultdict(dict)
    for seed, path in sorted(summaries.items()):
        count = 0
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"variable", "layer", "cal_iia"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing columns {required} in {path}")

            for row in reader:
                variable = row["variable"].strip().upper()
                if variable not in VARIABLES:
                    continue
                layer = int(row["layer"])
                score = float(row["cal_iia"])
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"cal_iia must be in [0,1]: {path}, {variable} L{layer}")
                key = (variable, layer)
                if seed in values[key]:
                    raise ValueError(f"Duplicate {variable} L{layer} in seed {seed}")
                values[key][seed] = score
                count += 1
        print(f"[read] seed={seed}: {count} CAL IIA rows from {path}")
    return values


def summarize(values, all_seeds, strict):
    result = []
    expected = set(all_seeds)
    for variable, layer in sorted(values, key=lambda x: (VARIABLES.index(x[0]), x[1])):
        by_seed = values[(variable, layer)]
        missing = sorted(expected - set(by_seed))
        if missing:
            message = f"{variable} L{layer}: missing seeds {missing} (using {len(by_seed)}/{len(all_seeds)})"
            if strict:
                raise ValueError(message)
            print(f"[warning] {message}")

        scores = list(by_seed.values())
        avg = mean(scores) * 100
        sd = stdev(scores) * 100 if len(scores) > 1 else float("nan")
        result.append({
            "variable": variable,
            "layer": layer,
            "n_seeds": len(scores),
            "seeds": ",".join(map(str, sorted(by_seed))),
            "mean_cal_iia_pct": avg,
            "std_cal_iia_pct": sd,
        })

    if strict:
        for variable in VARIABLES:
            if not any(row["variable"] == variable for row in result):
                raise ValueError(f"No rows for {variable}")
    return result


def save_csv(rows, output):
    fields = ["variable", "layer", "n_seeds", "seeds", "mean_cal_iia_pct", "std_cal_iia_pct"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_annotation(row, total_seeds):
    avg = row["mean_cal_iia_pct"]
    sd = row["std_cal_iia_pct"]
    sd_text = f"{sd:.1f}" if not np.isnan(sd) else "NA"
    text = f"{avg:.1f} ± {sd_text}"
    if row["n_seeds"] < total_seeds:
        text += f"\n(n={row['n_seeds']})"
    return text


def plot_variable(variable, rows, output_dir, total_seeds, dpi):
    selected = [row for row in rows if row["variable"] == variable]
    if not selected:
        print(f"[skip] No data for {variable}")
        return

    # One vertical column: layer increases downward on the Y axis.
    layers = [row["layer"] for row in selected]
    means = np.array([row["mean_cal_iia_pct"] for row in selected], dtype=float)
    height = max(5.0, 0.43 * len(layers) + 1.5)
    fig, ax = plt.subplots(figsize=(4.7, height))
    im = ax.imshow(means.reshape(-1, 1), aspect="auto", cmap="viridis", vmin=0, vmax=100)

    ax.set_xticks([0], labels=[variable])
    ax.set_yticks(np.arange(len(layers)), labels=[f"L{layer}" for layer in layers])
    ax.set_ylabel("Layer")
    ax.set_title(f"{variable} CAL IIA — mean ± std (%)", pad=12)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", left=False, bottom=False)

    for i, row in enumerate(selected):
        avg = row["mean_cal_iia_pct"]
        ax.text(0, i, format_annotation(row, total_seeds),
                ha="center", va="center", fontsize=8,
                color="black" if avg >= 60 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.06)
    cbar.set_label("Mean CAL IIA (%)")
    fig.tight_layout()
    output = output_dir / f"{variable}_cal_iia_heatmap.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def plot_combined(rows, output_dir, total_seeds, dpi):
    # Three vertical strips (AT, AP, C) with a common layer axis.
    layers = sorted({row["layer"] for row in rows})
    if not layers:
        return
    lookup = {(row["variable"], row["layer"]): row for row in rows}
    data = np.full((len(layers), len(VARIABLES)), np.nan, dtype=float)

    for i, layer in enumerate(layers):
        for j, variable in enumerate(VARIABLES):
            entry = lookup.get((variable, layer))
            if entry is not None:
                data[i, j] = entry["mean_cal_iia_pct"]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e9ee")
    height = max(5.0, 0.43 * len(layers) + 1.5)
    fig, ax = plt.subplots(figsize=(9.0, height))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(VARIABLES)), labels=VARIABLES)
    ax.set_yticks(np.arange(len(layers)), labels=[f"L{layer}" for layer in layers])
    ax.set_ylabel("Layer")
    ax.set_title("MCQA CAL IIA — mean ± std across seeds (%)", pad=12)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(-0.5, len(VARIABLES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", left=False, bottom=False)

    for i, layer in enumerate(layers):
        for j, variable in enumerate(VARIABLES):
            entry = lookup.get((variable, layer))
            if entry is None:
                continue
            avg = entry["mean_cal_iia_pct"]
            ax.text(j, i, format_annotation(entry, total_seeds),
                    ha="center", va="center", fontsize=8,
                    color="black" if avg >= 60 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean CAL IIA (%)")
    fig.tight_layout()
    output = output_dir / "all_variables_cal_iia_heatmap.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Folder containing seed result directories")
    parser.add_argument("--pattern", default="results_standard_all_layers_mixedall_seed*_no_distractor")
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional seed filter, e.g. --seeds 0 1 2 3 4")
    parser.add_argument("--output-dir", type=Path, default=Path("cal_iia_heatmaps"))
    parser.add_argument("--strict", action="store_true", help="Fail if any selected seed lacks a variable/layer")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    summaries = find_summaries(args.root, args.pattern, args.seeds)
    print(f"[seeds] {sorted(summaries)}")
    values = load_cal_iia(summaries)
    rows = summarize(values, sorted(summaries), args.strict)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "cal_iia_mean_std.csv"
    save_csv(rows, csv_path)
    print(f"[saved] {csv_path}")
    for variable in VARIABLES:
        plot_variable(variable, rows, args.output_dir, len(summaries), args.dpi)
    plot_combined(rows, args.output_dir, len(summaries), args.dpi)


if __name__ == "__main__":
    main()
