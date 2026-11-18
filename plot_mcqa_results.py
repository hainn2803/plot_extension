#!/usr/bin/env python3
"""Plot selected MCQA Gradual DAS evaluation metrics (no model/GPU needed).

Figures (grouped by intervention):
  - AT_intervention.pdf: Joint (page 1), Readout and IIA (page 2)
  - AP_intervention.pdf: Joint (page 1), Readout/IIA/Replay/Restoration (page 2)
  - C_intervention.pdf: Joint (page 1), then six non-Joint metric pages
  - png/: one high-resolution image per PDF page
  - best_per_metric.csv: best observed layer configuration for each metric

All scores are displayed using the FIXED [0, 1] scale.

Usage:
  python3 plot_mcqa_eval_results.py --eval-dir results_.../eval_configs_gradual_<hash>
  python3 plot_mcqa_eval_results.py --eval-dir results_...  # newest eval_configs_gradual_*
  python3 plot_mcqa_eval_results.py --eval-dir ... --no-png

Inputs:
  config_summary.csv, or AT_L*__AP_L*__C_L*/metrics.json as fallback.
  Missing configurations are left blank, never interpolated.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless nodes / Slurm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np

# These match the exact keys written by mcqa_eval_gradual_completed.py.
JOINT = {
    "AT": "Joint_AT_Y_after_AT",
    "AP": "Joint_AP_AT_Y_after_AP",
    "C": "Joint_C_AP_AT_Y_after_C",
}
NONJOINT = {
    "AT": ["AT_after_AT", "Y_after_AT"],
    "AP": ["AP_after_AP", "Y_after_AP", "Y_via_AT_replay", "Y_restored_via_AT"],
    "C": [
        "C_after_C", "Y_after_C", "Y_via_AP_replay",
        "Y_via_AT_replay", "Y_restored_via_AP", "Y_restored_via_AT",
    ],
}

SPECS = []
for intervention in ("AT", "AP", "C"):
    SPECS.append((intervention, JOINT[intervention]))
    for name in NONJOINT[intervention]:
        SPECS.append((intervention, name))

DIMS = {
    "AT": ("at_layer",),
    "AP": ("at_layer", "ap_layer"),
    "C": ("at_layer", "ap_layer", "c_layer"),
}
LAYERS = ("at_layer", "ap_layer", "c_layer")
NORM = Normalize(vmin=0.0, vmax=1.0, clip=True)
COLORMAP = plt.get_cmap("Blues").copy()
COLORMAP.set_bad("#eceef1")


def find_eval_dir(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if (folder / "config_summary.csv").is_file():
        return folder
    if any(folder.glob("AT_L*__AP_L*__C_L*/metrics.json")):
        return folder

    candidates = []
    for child in folder.glob("eval_configs_gradual_*"):
        if not child.is_dir():
            continue
        has_csv = (child / "config_summary.csv").is_file()
        has_json = any(child.glob("AT_L*__AP_L*__C_L*/metrics.json"))
        if has_csv or has_json:
            candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No eval_configs_gradual_* results found under {folder}. "
            "Pass --eval-dir pointing at the evaluation output folder."
        )
    chosen = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    if len(candidates) > 1:
        print(f"[info] Found {len(candidates)} evaluation folders; using {chosen.name}")
    return chosen


def read_csv_records(file):
    records = []
    with file.open(newline="") as handle:
        for row in csv.DictReader(handle):
            record = {}
            for key, value in row.items():
                if value is None or not str(value).strip():
                    continue
                if key in LAYERS:
                    record[key] = int(value)
                else:
                    try:
                        record[key] = float(value)
                    except ValueError:
                        continue
            records.append(record)
    return records


def read_json_records(folder):
    records = []
    for file in sorted(folder.glob("AT_L*__AP_L*__C_L*/metrics.json")):
        saved = json.loads(file.read_text())
        record = {}
        for layer in LAYERS:
            record[layer] = int(saved[layer])
        for variable, metrics in saved.get("metrics", {}).items():
            for name, value in metrics.items():
                record[f"{variable}_{name}"] = float(value)
        records.append(record)
    return records


def load_records(folder):
    csv_file = folder / "config_summary.csv"
    records = read_csv_records(csv_file) if csv_file.is_file() else []
    expected = {f"{variable}_{name}" for variable, name in SPECS}
    columns = set().union(*(set(row) for row in records)) if records else set()

    if not records or not expected.issubset(columns):
        json_records = read_json_records(folder)
        if json_records:
            if records:
                print("[info] CSV does not contain all selected metrics; reading metrics.json instead")
            records = json_records
        elif not records:
            raise ValueError("No records found in config_summary.csv or metrics.json")

    seen = set()
    for row in records:
        if any(layer not in row for layer in LAYERS):
            raise ValueError("Missing at_layer/ap_layer/c_layer in input")
        key = tuple(row[layer] for layer in LAYERS)
        if key in seen:
            raise ValueError(f"Duplicate configuration {key}")
        seen.add(key)

    records.sort(key=lambda row: tuple(row[layer] for layer in LAYERS))
    present = {f"{var}_{name}" for var, name in SPECS if any(f"{var}_{name}" in row for row in records)}
    missing = sorted(expected - present)
    if missing:
        print(f"[warning] {len(missing)} selected metrics missing; affected plots will show N/A")
        for name in missing:
            print(f"  - {name}")
    print(f"[load] {len(records)} completed configurations from {folder}")
    return records


def all_layers(records, field):
    return sorted({row[field] for row in records})


def aggregated(records, variable, name):
    """AT metrics independent of AP/C; AP metrics independent of C."""
    field = f"{variable}_{name}"
    groups = defaultdict(list)
    for row in records:
        value = row.get(field, math.nan)
        if not math.isfinite(value):
            continue
        key = tuple(row[d] for d in DIMS[variable])
        groups[key].append(value)

    values = {}
    inconsistent = []
    for key, scores in groups.items():
        if max(scores) - min(scores) > 1e-5:
            inconsistent.append(key)
        values[key] = float(np.mean(scores))
    if inconsistent:
        print(f"[warning] {field}: {len(inconsistent)} repeated layer groups disagree; using mean")
    return values


def matrix_2d(values, rows, cols, make_key):
    matrix = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for iy, row_layer in enumerate(rows):
        for ix, col_layer in enumerate(cols):
            matrix[iy, ix] = values.get(make_key(row_layer, col_layer), np.nan)
    return matrix


def text_color(score):
    r, g, b, _ = COLORMAP(NORM(score))
    luminance = .2126 * r + .7152 * g + .0722 * b
    return "#15202b" if luminance > 0.54 else "#ffffff"


def draw_heatmap(ax, matrix, x_layers, y_layers, title, x_label, y_label,
                 annot=True, font_size=8):
    mask = np.ma.masked_invalid(matrix)
    image = ax.imshow(mask, cmap=COLORMAP, norm=NORM,
                      aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=12, pad=10, fontweight="semibold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(x_layers)), [f"L{v}" for v in x_layers])
    ax.set_yticks(range(len(y_layers)), [f"L{v}" for v in y_layers])
    ax.tick_params(length=0, labelsize=9)
    ax.set_xticks(np.arange(-.5, len(x_layers), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(y_layers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annot:
        for iy in range(matrix.shape[0]):
            for ix in range(matrix.shape[1]):
                val = matrix[iy, ix]
                if np.isfinite(val):
                    ax.text(ix, iy, f"{val:.3f}", ha="center", va="center",
                            fontsize=font_size, color=text_color(val))
    return image


def shared_colorbar(fig, axes, image):
    fig.colorbar(image, ax=list(np.ravel(axes)), shrink=.84, pad=.015,
                 ticks=[0, .2, .4, .6, .8, 1.0], label="Accuracy (0–1)")


def draw_at(ax, records, metric):
    values = aggregated(records, "AT", metric)
    xs = all_layers(records, "at_layer")
    ys = [values.get((layer,), np.nan) for layer in xs]
    ax.plot(xs, ys, color="#2365a5", linewidth=2.4, marker="o", markersize=7)
    ax.set_title(metric, fontweight="semibold", fontsize=12)
    ax.set_xlabel("AT layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            offset = -15 if y >= .96 else 9
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, offset),
                        textcoords="offset points", ha="center", fontsize=9)


def draw_ap(ax, records, metric, title=None):
    at_layers = all_layers(records, "at_layer")
    ap_layers = all_layers(records, "ap_layer")
    values = aggregated(records, "AP", metric)
    matrix = matrix_2d(values, ap_layers, at_layers,
                       lambda ap, at: (at, ap))
    return draw_heatmap(ax, matrix, at_layers, ap_layers, title or metric,
                        "AT layer", "AP layer", font_size=8)


def draw_c_panels(fig, axes, records, metric):
    at_layers = all_layers(records, "at_layer")
    ap_layers = all_layers(records, "ap_layer")
    # c_layers = all_layers(records, "c_layer")
    c_layers = [layer for layer in all_layers(records, "c_layer") if 0 <= layer <= 1]
    values = aggregated(records, "C", metric)
    image = None
    for ax, at in zip(axes, at_layers):
        matrix = matrix_2d(values, ap_layers, c_layers,
                           lambda ap, c: (at, ap, c))
        image = draw_heatmap(ax, matrix, c_layers, ap_layers,
                             f"AT = L{at}", "C layer", "AP layer", font_size=8)
    for ax in axes[len(at_layers):]:
        ax.set_visible(False)
    if image is not None:
        shared_colorbar(fig, axes[:len(at_layers)], image)


def style_fig(fig, title, subtitle=None):
    fig.suptitle(title, fontsize=17, fontweight="bold")
    if subtitle:
        fig.text(.5, .965, subtitle, ha="center", va="top",
                 fontsize=10, color="#5e6a77")


def fig_at_joint(records):
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    draw_at(ax, records, JOINT["AT"])
    style_fig(fig, "AT intervention | Joint")
    return fig


def fig_ap_joint(records):
    aps = len(all_layers(records, "ap_layer"))
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, .49 * aps + 2.5)), layout="constrained")
    im = draw_ap(ax, records, JOINT["AP"])
    shared_colorbar(fig, [ax], im)
    style_fig(fig, "AP intervention | Joint")
    return fig


def fig_c_metric(records, metric, is_joint=False):
    ats = all_layers(records, "at_layer")
    aps = all_layers(records, "ap_layer")
    # Usually 3 AT layers; for more, use multiple rows automatically.
    ncol = min(len(ats), 4)
    nrow = math.ceil(len(ats) / ncol)
    width = max(11.5, 4.2 * ncol + 1.0)
    height = max(7.0, (max(4.8, .38 * len(aps) + 1.6)) * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, height),
                             layout="constrained", squeeze=False)
    draw_c_panels(fig, list(axes.flat), records, metric)
    label = "C intervention | Joint" if is_joint else "C intervention"
    style_fig(fig, label + f"\n{metric}")
    return fig


def fig_at_nonjoint(records):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), layout="constrained")
    for ax, metric in zip(axes, NONJOINT["AT"]):
        draw_at(ax, records, metric)
    style_fig(fig, "AT intervention | Readout and IIA")
    return fig


def fig_ap_nonjoint(records):
    aps = len(all_layers(records, "ap_layer"))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, max(10.0, .85 * aps + 4.5)),
                             layout="constrained")
    last_image = None
    for ax, metric in zip(axes.flat, NONJOINT["AP"]):
        last_image = draw_ap(ax, records, metric)
    shared_colorbar(fig, axes, last_image)
    style_fig(fig, "AP intervention | Readout, IIA, Replay, Restoration")
    return fig


def write_best(records, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "intervention", "metric", "best_score", "at_layer", "ap_layer", "c_layer"])
        writer.writeheader()
        for variable, metric in SPECS:
            vals = aggregated(records, variable, metric)
            if not vals:
                continue
            key, value = max(vals.items(), key=lambda item: item[1])
            row = {
                "intervention": variable,
                "metric": metric,
                "best_score": f"{value:.6f}",
                "at_layer": key[0],
                "ap_layer": key[1] if len(key) > 1 else "",
                "c_layer": key[2] if len(key) > 2 else "",
            }
            writer.writerow(row)


def save_bundle(pdf_path, fig_specs, png_dir, no_png, dpi):
    print(f"[pdf] {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        for filename, maker in fig_specs:
            fig = maker()
            pdf.savefig(fig, facecolor="white")
            if not no_png:
                dest = png_dir / f"{filename}.png"
                fig.savefig(dest, dpi=dpi, facecolor="white")
                print(f"[png] {dest.name}")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", required=True,
                        help="Folder eval_configs_gradual_<hash> or its parent")
    parser.add_argument("--out-dir", default=None,
                        help="Default: <eval-dir>/figures")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    folder = find_eval_dir(args.eval_dir)
    records = load_records(folder)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else folder / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    if not args.no_png:
        png_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,  # keep labels as editable vector text where supported
        "savefig.facecolor": "white",
    })

    figures_by_intervention = {
        "AT": [
            ("AT_joint", lambda: fig_at_joint(records)),
            ("AT_readout_IIA", lambda: fig_at_nonjoint(records)),
        ],
        "AP": [
            ("AP_joint", lambda: fig_ap_joint(records)),
            ("AP_readout_IIA_replay_restoration", lambda: fig_ap_nonjoint(records)),
        ],
        "C": [
            ("C_joint", lambda: fig_c_metric(records, JOINT["C"], True)),
        ],
    }
    for metric in NONJOINT["C"]:
        figures_by_intervention["C"].append(
            (f"C_{metric}", lambda metric=metric: fig_c_metric(records, metric))
        )

    for variable in ("AT", "AP", "C"):
        save_bundle(
            out_dir / f"{variable}_intervention.pdf",
            figures_by_intervention[variable],
            png_dir,
            args.no_png,
            args.dpi,
        )
    write_best(records, out_dir / "best_per_metric.csv")
    print(f"[done] {out_dir} | 15 metrics grouped by intervention in 3 PDFs (2 + 2 + 7 pages)")


if __name__ == "__main__":
    main()
