#!/usr/bin/env python3
"""
Figure 7 -- NanoGPT spectral entropy.

(a) Mean spectral entropy across all weight layers during training: Muon vs AdamW.
    Lines show mean over seeds; shading shows seed-to-seed spread.
(b) Per-layer spectral entropy at convergence (grouped bar chart).

Reads: results/nanogpt/01_muon_vs_adamw/*/spectral/spectral_log.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    from plots.style import setup_style, get_color, save_fig, add_panel_label, DOUBLE_COL
except ImportError:
    from style import setup_style, get_color, save_fig, add_panel_label, DOUBLE_COL


def _load_spectral_runs(exp_dir: Path) -> dict:
    """
    Load spectral_log.jsonl from each run in 01_muon_vs_adamw.

    Returns: {
        optimizer: {
            "seeds": [
                {
                    "iters": [int, ...],
                    "mean_entropy_per_iter": [float, ...],   # mean across layers
                    "per_layer_final": {layer_name: entropy, ...}
                },
                ...
            ]
        }
    }
    """
    results = defaultdict(lambda: {"seeds": []})

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        spectral_path = run_dir / "spectral" / "spectral_log.jsonl"
        if not spectral_path.exists():
            continue

        # Determine optimizer from summary.json
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        opt = summary["optimizer"]

        # Parse spectral log
        iters = []
        mean_entropy_per_iter = []
        last_snapshot = None

        with open(spectral_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                it = record["iter"]
                layers = record["layers"]

                # Mean spectral entropy across all layers at this iteration
                entropies = [layer["spectral_entropy"] for layer in layers]
                iters.append(it)
                mean_entropy_per_iter.append(np.mean(entropies))
                last_snapshot = record

        # Per-layer entropy at convergence (final snapshot)
        per_layer_final = {}
        if last_snapshot is not None:
            for layer in last_snapshot["layers"]:
                per_layer_final[layer["name"]] = layer["spectral_entropy"]

        results[opt]["seeds"].append({
            "iters": iters,
            "mean_entropy_per_iter": mean_entropy_per_iter,
            "per_layer_final": per_layer_final,
        })

    return results


def plot(results_dir: str, output_dir: str):
    setup_style()
    exp_dir = Path(results_dir) / "nanogpt" / "01_muon_vs_adamw"
    data = _load_spectral_runs(exp_dir)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    opt_pairs = [("muon", "Muon"), ("adamw", "AdamW")]

    # ------------------------------------------------------------------
    # (a) Mean spectral entropy over training (averaged across seeds)
    # ------------------------------------------------------------------
    for opt_key, display in opt_pairs:
        if opt_key not in data or len(data[opt_key]["seeds"]) == 0:
            continue

        seeds = data[opt_key]["seeds"]
        # All seeds should share the same iteration grid
        ref_iters = np.array(seeds[0]["iters"])

        # Stack mean entropy trajectories across seeds
        all_curves = []
        for s in seeds:
            curve = np.array(s["mean_entropy_per_iter"])
            # Align lengths (take min)
            n = min(len(ref_iters), len(curve))
            all_curves.append(curve[:n])

        n = min(len(c) for c in all_curves)
        ref_iters = ref_iters[:n]
        all_curves = np.array([c[:n] for c in all_curves])

        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)

        ax_a.plot(ref_iters, mean_curve, color=get_color(display), label=display)
        ax_a.fill_between(ref_iters, mean_curve - std_curve, mean_curve + std_curve,
                          color=get_color(display), alpha=0.15)

    ax_a.set_xlabel("Training step")
    ax_a.set_ylabel("Mean spectral entropy $H$")
    ax_a.legend(loc="lower right")
    add_panel_label(ax_a, "a")

    # ------------------------------------------------------------------
    # (b) Per-layer entropy at convergence (grouped bar chart)
    # ------------------------------------------------------------------
    # Collect layer names from first available seed
    ref_opt = None
    layer_names = []
    for opt_key, _ in opt_pairs:
        if opt_key in data and len(data[opt_key]["seeds"]) > 0:
            ref_opt = opt_key
            layer_names = list(data[opt_key]["seeds"][0]["per_layer_final"].keys())
            break
    if not layer_names:
        save_fig(fig, "fig7_nanogpt_spectral", output_dir)
        return

    n_layers = len(layer_names)
    x = np.arange(n_layers)
    width = 0.35

    for i, (opt_key, display) in enumerate(opt_pairs):
        if opt_key not in data or len(data[opt_key]["seeds"]) == 0:
            continue
        seeds = data[opt_key]["seeds"]
        # Average per-layer entropy across seeds
        per_layer_means = []
        per_layer_stds = []
        for lname in layer_names:
            vals = [s["per_layer_final"].get(lname, np.nan) for s in seeds]
            per_layer_means.append(np.nanmean(vals))
            per_layer_stds.append(np.nanstd(vals))
        per_layer_means = np.array(per_layer_means)
        per_layer_stds = np.array(per_layer_stds)

        offset = (i - 0.5) * width
        ax_b.bar(x + offset, per_layer_means, width, yerr=per_layer_stds,
                 color=get_color(display), edgecolor="black", linewidth=0.3,
                 capsize=1.5, error_kw={"linewidth": 0.6}, label=display)

    ax_b.set_xticks(x)
    short_names = [n.replace("transformer.", "").replace(".weight", "")
                   for n in layer_names]
    ax_b.set_xticklabels(short_names, rotation=55, ha="right", fontsize=5)
    ax_b.set_ylabel("Spectral entropy $H$")
    ax_b.legend(loc="upper right")
    add_panel_label(ax_b, "b")

    save_fig(fig, "fig7_nanogpt_spectral", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 7 -- NanoGPT spectral")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
