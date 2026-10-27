"""Evidence-backed reports. Missing stages are reported as missing, never successful."""
from pathlib import Path
import csv
import numpy as np
from scipy.stats import spearmanr

from .protocol import COLORS, LABELS
from .storage import read, write, file_hash
from .metrics import paired_group_bootstrap
from .metrics import discrete_gate, structural_gates


def localization_figures(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    rankings = read(root / "localization/rankings.json")
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    for variable, methods in rankings.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        sorted_scores = {}
        for axis, (method, rows) in zip(axes, methods.items()):
            width = max(r["position"] for r in rows)
            matrix = np.full((26, width), np.nan)
            for row in rows:
                matrix[row["layer"], row["position"]-1] = row["score"]
            sorted_scores[method] = matrix.ravel()
            plot = axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest",
                               extent=(.5, width+.5, -.5, 25.5))
            axis.set_title(variable + ": " + method)
            axis.set_xlabel("Fixed token index (excluding BOS)")
            axis.set_ylabel("Decoder block index")
            fig.colorbar(plot, ax=axis)
        fig.savefig(directory / f"{variable}_localization.png", dpi=150)
        plt.close(fig)
        comparisons[variable] = {}
        names = list(methods)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                top_a = {(r["layer"], r["position"]) for r in methods[a][:10]}
                top_b = {(r["layer"], r["position"]) for r in methods[b][:10]}
                comparisons[variable][a + "__" + b] = {
                    "spearman": float(spearmanr(sorted_scores[a], sorted_scores[b]).statistic),
                    "top_ten_overlap": len(top_a & top_b)}
    write(root / "localization/agreement.json", comparisons, immutable=True)
    return comparisons


def training_outputs(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    histories = {p.parent.name: read(p) for p in sorted((root / "training").glob("*/history.json"))}
    if not histories:
        return
    fields = ["variable", "layer", "position", "rank", "seed", "epoch", "cal_accuracy", "cal_ce",
              "train_online_changing_accuracy", "train_online_preservation_accuracy", "train_ce", "gradient_norm_mean", "seconds"]
    with (root / "calibration_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rows in histories.values():
            writer.writerows(rows)
    if not (root / "selections/chain_seeds.json").exists():
        return
    from .evaluation import handle_name
    selection = read(root / "selections/seed1.json")
    chains = read(root / "selections/chain_seeds.json")
    chosen = {handle_name(c): c for methods in selection["method_winners"].values() for c in methods.values()}
    chosen.update({handle_name(c): c for pair in chains.values() if pair is not None for c in pair.values()})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row, variable in enumerate(("C", "AT")):
        for name, candidate in sorted(chosen.items()):
            if candidate["variable"] != variable:
                continue
            history = histories[name]
            for axis, metric in zip(axes[row], ("cal_accuracy", "train_online_changing_accuracy", "train_online_preservation_accuracy")):
                axis.plot([r["epoch"] for r in history], [r[metric] for r in history], label=name, linewidth=1.2)
                axis.set(ylim=(0, 1.02), xlabel="Epoch", ylabel="Accuracy", title=variable + ": " + metric)
        axes[row, 0].legend(fontsize=7)
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "selected_learning_curves.png", dpi=150)
    plt.close(fig)


def method_comparisons(root):
    from .evaluation import handle_name
    selection = read(root / "selections/seed1.json")
    rows = read(root / "banks/test.json")
    result = {}
    for variable, methods in selection["method_winners"].items():
        names = sorted(methods)
        for i, left in enumerate(names):
            a = read(root / "evaluation/direct" / (handle_name(methods[left]) + ".json"))
            for right in names[i+1:]:
                b = read(root / "evaluation/direct" / (handle_name(methods[right]) + ".json"))
                if a["pair_ids"] != b["pair_ids"] or a["pair_ids"] != [r["id"] for r in rows]:
                    raise ValueError("Method comparison is not paired on identical examples")
                values = {}
                for kind, mask in (("changing", np.array([r[variable+"_changed"] for r in rows])),
                                   ("preservation", np.array([not r[variable+"_changed"] for r in rows]))):
                    values[kind] = paired_group_bootstrap([r["group"] for r, yes in zip(rows, mask) if yes],
                        np.array(a["correct"])[mask], np.array(b["correct"])[mask])
                result[variable + ":" + left + "-" + right] = values
    write(root / "evaluation/method_comparisons.json", result, immutable=True)


def report(root):
    root = Path(root)
    training_outputs(root)
    lines = ["# C -> AT -> Y: Experiment Status", ""]
    profile_path = root / "engineering/profile.json"
    if profile_path.exists():
        profile = read(profile_path)
        lines += ["## Local Engineering", "",
                  f"Largest validated dedicated-memory batch: {profile['selected_batch']}.",
                  f"Conservative maximum-schedule GPU estimate including 25% allowance: {profile['hours_with_25_percent_allowance']:.2f} hours.",
                  "This is a bound using early-layer rank-512 timings, not a measured complete run.",
                  f"Local launch authorized: {profile['launch_allowed']}. Sustained thermal gate: {profile['sustained_thermal_gate']}.", ""]
    if not (root / "registration.json").exists():
        lines += ["## Scientific Execution", "", "Not registered or started. No filtering, localization, training, or heldout results.", ""]
    else:
        registration = read(root / "registration.json")
        lines += ["## Banks", "", f"One-color requested-object semantics: {registration['one_color_query_mode']}.", ""]
        for bank in registration["proposals"]:
            path = root / "banks" / f"{bank}.json"
            lines.append(f"- {bank}: {len(read(path))} retained records." if path.exists() else f"- {bank}: proposed only; no retained-bank result.")
    if (root / "localization/rankings.json").exists():
        localization_figures(root)
        lines += ["", "## Localization", "", "Complete maps and agreement checks are in figures/ and localization/.",
                  "PLOT and full-vector scores share swap measurements; their agreement is not independent causal evidence."]
    if (root / "selections/seed1.json").exists():
        selection = read(root / "selections/seed1.json")
        lines += ["", "## Calibration-Selected Handles", "", "Layers are zero-indexed decoder blocks; token positions include BOS at index 0.",
                  "", "| Variable | Localization | Block | Token | Rank | Epoch | Calibration |", "|---|---|---:|---:|---:|---:|---:|"]
        for variable, methods in selection["method_winners"].items():
            for method, c in methods.items():
                lines.append(f"| {variable} | {method} | {c['layer']} | {c['position']} | {c['rank']} | {c['epoch']} | {c['cal_accuracy']:.3f} |")
        if selection["chain"] is None:
            lines += ["", "No chain-compatible C/AT pair was found in the registered shortlist; no additional search or seed repeats were substituted."]
    direct = sorted((root / "evaluation/direct").glob("*.json")) if (root / "evaluation/direct").exists() else []
    if direct:
        from .evaluation import handle_name
        selection = read(root / "selections/seed1.json")
        if all((root / "evaluation/direct" / (handle_name(c) + ".json")).exists()
               for methods in selection["method_winners"].values() for c in methods.values()):
            method_comparisons(root)
        lines += ["", "## Heldout Direct Interventions", "", "| Handle | Changing | Preservation | Gate |", "|---|---:|---:|---|"]
        for path in direct:
            result = read(path)
            summary = result["summary"]
            lines.append(f"| {path.stem} | {summary['changing']:.3f} | {summary['preservation']:.3f} | {result['gate']['passed']} |")
    chain_files = sorted((root / "evaluation").glob("chain_seed_*.json")) if (root / "evaluation").exists() else []
    if chain_files:
        from .evaluation import handle_name
        reliability = read(root / "evaluation/readout_reliability.json")
        decisions = {}
        all_c, all_chain = [], []
        for path in chain_files:
            result = read(path)
            pair = result["pair"]
            direct_results = {v: read(root / "evaluation/direct" / (handle_name(c) + ".json")) for v, c in pair.items()}
            structures = structural_gates(result["structural"])
            replay = {k: discrete_gate(summary) for k, summary in result["summary"].items()}
            readout_pass = all(reliability[handle_name(c)]["validation_macro_accuracy"] >= .9 and
                              min(reliability[handle_name(c)]["validation_by_class"].values()) >= .9 for c in pair.values())
            c_pass = direct_results["C"]["gate"]["passed"]
            chain_pass = c_pass and direct_results["AT"]["gate"]["passed"] and readout_pass and all(g["passed"] for g in replay.values()) and all(g["passed"] for g in structures.values())
            decisions[str(pair["C"]["seed"])] = {"C_handle_gate": c_pass, "chain_gate": chain_pass,
                "AT_handle_gate": direct_results["AT"]["gate"]["passed"],
                "readout_reliability_gate": readout_pass, "replay_gates": replay, "structural_gates": structures}
            all_c.append(c_pass)
            all_chain.append(chain_pass)
        decisions["across_all_three_seeds"] = {"C_handle": len(all_c) == 3 and all(all_c), "chain_support": len(all_chain) == 3 and all(all_chain)}
        write(root / "evaluation/decisions.json", decisions, immutable=True)
        lines += ["", "## Three-Seed Decision", "", f"C-handle gates passed in all three seeds: {decisions['across_all_three_seeds']['C_handle']}.",
                  f"Chain gates passed in all three seeds: {decisions['across_all_three_seeds']['chain_support']}.",
                  "Detailed failures and distinct-case counts are in evaluation/decisions.json."]
        controls_reliable = all(d["readout_reliability_gate"] and d["AT_handle_gate"] and
            all(d["replay_gates"][k]["passed"] for k in ("recipient_change_clean_donor", "recipient_preserve_clean_donor"))
            for seed, d in decisions.items() if seed != "across_all_three_seeds")
        if not decisions["across_all_three_seeds"]["C_handle"]:
            recommendation = "Gradual DAS is not yet justified: first establish a C handle that passes the declared transfer and preservation gates."
        elif not controls_reliable:
            recommendation = "Resolve weak readouts or AT donor controls before attributing replay failures to the C handle or training gradual DAS."
        elif decisions["across_all_three_seeds"]["chain_support"]:
            recommendation = "Standard DAS passes the declared chain tests; these results do not yet demonstrate an internal deficit for gradual DAS to fix."
        else:
            recommendation = "A preregistered gradual-DAS follow-up is worth considering for the remaining behavioral/internal deficits, with the reliable AT controls retained. No improvement is assumed."
        lines += ["", "## Gradual-DAS Decision", "", recommendation]
    if (root / "runtime_accounting.json").exists():
        accounting = read(root / "runtime_accounting.json")
        lines += ["", "## Actual Runtime", "", "| Stage | Hours |", "|---|---:|"]
        for name, value in sorted(accounting["stages"].items()):
            lines.append(f"| {name} | {value['seconds']/3600:.3f} |")
        lines += ["", f"Completed-stage total: {accounting['total_stage_seconds']/3600:.2f} hours. Interrupted attempts are preserved separately."]
    lines += ["", "## Claim Limits", "", "Missing measurements or coverage are not passing results. Full-chain certification requires all registered structural, donor, readout, and directional gates.",
              "There are only 16 final two-object groups. Seeds share that bank and are not independent data replications.",
              "No result establishes unrestricted color understanding or the absence of AP.",
              "A subsequent gradual-DAS experiment remains deferred pending the completed causal evaluation.", ""]
    path = root / "STATUS_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
