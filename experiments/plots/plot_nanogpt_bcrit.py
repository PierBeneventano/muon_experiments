#!/usr/bin/env python3
"""
Figure 6 — NanoGPT critical batch size.

(a) Final validation loss vs batch size for Muon and AdamW (identifying B_crit).
(b) Optimal LR vs batch size for Muon and AdamW (showing LR plateau region).

Reads:
    results/nanogpt/02_batch_size_sweep/
    results/nanogpt/03_lr_sweep/
"""

import argparse
import json
import glob as globmod
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from plots.style import setup_style, get_color, get_marker, save_fig, add_panel_label, DOUBLE_COL
except ImportError:
    from style import setup_style, get_color, get_marker, save_fig, add_panel_label, DOUBLE_COL


def _collect_sweep(sweep_dir: str) -> dict:
    """
    Scan *sweep_dir* for JSON result files and collect per-optimizer data.
    Returns: {optimizer: {"batch_sizes": [...], "val_losses": [...], "best_lrs": [...]}}
    """
    results = {}
    for p in sorted(Path(sweep_dir).glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        opt = d["optimizer"]
        if opt not in results:
            results[opt] = {"batch_sizes": [], "val_losses": [], "best_lrs": []}
        results[opt]["batch_sizes"].append(d["batch_size"])
        results[opt]["val_losses"].append(d["final_val_loss"])
        if "best_lr" in d:
            results[opt]["best_lrs"].append(d["best_lr"])
    return results


def plot(results_dir: str, output_dir: str):
    setup_style()

    bs_dir = Path(results_dir) / "nanogpt" / "02_batch_size_sweep"
    lr_dir = Path(results_dir) / "nanogpt" / "03_lr_sweep"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # ------------------------------------------------------------------
    # (a) Val loss vs batch size
    # ------------------------------------------------------------------
    bs_data = _collect_sweep(str(bs_dir))
    for opt_name, display in [("muon", "Muon"), ("adamw", "AdamW")]:
        if opt_name not in bs_data:
            continue
        entry = bs_data[opt_name]
        # Aggregate across seeds: group by batch_size
        unique_bs = sorted(set(entry["batch_sizes"]))
        means, stds = [], []
        for bs in unique_bs:
            vals = [v for b, v in zip(entry["batch_sizes"], entry["val_losses"]) if b == bs]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)

        ax_a.errorbar(unique_bs, means, yerr=stds,
                      color=get_color(display), marker=get_marker(display),
                      label=display, capsize=2, linewidth=1.0)

    ax_a.set_xscale("log", base=2)
    ax_a.set_xlabel("Batch size")
    ax_a.set_ylabel("Final validation loss")
    ax_a.legend(loc="upper right")
    add_panel_label(ax_a, "a")

    # ------------------------------------------------------------------
    # (b) Optimal LR vs batch size
    # ------------------------------------------------------------------
    lr_data = _collect_sweep(str(lr_dir))
    for opt_name, display in [("muon", "Muon"), ("adamw", "AdamW")]:
        if opt_name not in lr_data:
            continue
        entry = lr_data[opt_name]
        unique_bs = sorted(set(entry["batch_sizes"]))
        best = []
        for bs in unique_bs:
            lrs = [v for b, v in zip(entry["batch_sizes"], entry["best_lrs"]) if b == bs]
            if lrs:
                best.append(np.mean(lrs))
            else:
                best.append(np.nan)
        ax_b.plot(unique_bs, best, color=get_color(display),
                  marker=get_marker(display), label=display)

    ax_b.set_xscale("log", base=2)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Batch size")
    ax_b.set_ylabel("Optimal learning rate")
    ax_b.legend(loc="upper left")
    add_panel_label(ax_b, "b")

    save_fig(fig, "fig6_nanogpt_bcrit", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 6 — NanoGPT B_crit")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
