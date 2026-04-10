#!/usr/bin/env python3
"""
Figure 6 -- NanoGPT critical batch size.

(a) Final validation loss vs batch size for Muon and AdamW (identifying B_crit).
(b) Optimal LR vs batch size for Muon and AdamW (showing LR plateau region).

Reads:
    results/nanogpt/02_batch_size_sweep/*/summary.json  (+ stdout.txt for val loss)
    results/nanogpt/03_lr_sweep/*/summary.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from plots.style import setup_style, get_color, get_marker, save_fig, add_panel_label, DOUBLE_COL
except ImportError:
    from style import setup_style, get_color, get_marker, save_fig, add_panel_label, DOUBLE_COL


def _parse_val_loss_from_stdout(stdout_path: Path) -> Optional[float]:
    """Parse the final val loss from stdout.txt ('step XXXX: train loss X.XXXX, val loss X.XXXX')."""
    if not stdout_path.exists():
        return None
    last_val = None
    pattern = re.compile(r"step\s+\d+:\s+train loss\s+[\d.]+,\s+val loss\s+([\d.]+)")
    with open(stdout_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last_val = float(m.group(1))
    return last_val


def _get_final_val_loss(run_dir: Path) -> Optional[float]:
    """Get final val loss from a run directory: try summary.json first, fall back to stdout.txt."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        d = json.load(f)
    # Check if summary has val_losses (exp01 format: plain list of floats)
    if "final_val_loss" in d and d["final_val_loss"] is not None:
        return float(d["final_val_loss"])
    if "val_losses" in d and isinstance(d["val_losses"], list) and len(d["val_losses"]) > 0:
        return float(d["val_losses"][-1])
    # Fall back to stdout parsing
    return _parse_val_loss_from_stdout(run_dir / "stdout.txt")


def _collect_batch_size_sweep(sweep_dir: Path) -> dict:
    """
    Scan 02_batch_size_sweep subdirectories for summary.json files.
    Returns: {optimizer: {batch_size: [val_losses across seeds]}}
    """
    results = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            d = json.load(f)
        opt = d["optimizer"]
        bs = d["batch_size"]
        val_loss = _get_final_val_loss(run_dir)
        if val_loss is not None:
            results[opt][bs].append(val_loss)
    return results


def _collect_lr_sweep(sweep_dir: Path) -> dict:
    """
    Scan 03_lr_sweep subdirectories for summary.json files.
    Returns: {optimizer: {batch_size: [(lr, final_train_loss)]}}
    """
    results = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            d = json.load(f)
        opt = d["optimizer"]
        lr = d["lr"]
        bs = d["batch_size"]
        # Get final train loss
        ftl = d.get("final_train_loss")
        if ftl is None:
            # Compute from train_losses list
            tl = d.get("train_losses", [])
            if tl:
                ftl = tl[-1]["loss"] if isinstance(tl[-1], dict) else tl[-1]
        if ftl is not None:
            results[opt][bs].append((lr, ftl))
    return results


def plot(results_dir: str, output_dir: str):
    setup_style()

    bs_dir = Path(results_dir) / "nanogpt" / "02_batch_size_sweep"
    lr_dir = Path(results_dir) / "nanogpt" / "03_lr_sweep"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # ------------------------------------------------------------------
    # (a) Val loss vs batch size
    # ------------------------------------------------------------------
    bs_data = _collect_batch_size_sweep(bs_dir)
    for opt_key, display in [("muon", "Muon"), ("adamw", "AdamW")]:
        if opt_key not in bs_data:
            continue
        bs_dict = bs_data[opt_key]
        batch_sizes = sorted(bs_dict.keys())
        means, stds = [], []
        for bs in batch_sizes:
            vals = np.array(bs_dict[bs])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)

        ax_a.errorbar(batch_sizes, means, yerr=stds,
                      color=get_color(display), marker=get_marker(display),
                      label=display, capsize=3, linewidth=1.0,
                      markeredgecolor="black", markeredgewidth=0.3)

    ax_a.set_xscale("log", base=2)
    ax_a.set_xlabel("Batch size")
    ax_a.set_ylabel("Final validation loss")
    ax_a.legend(loc="upper right")
    add_panel_label(ax_a, "a")

    # ------------------------------------------------------------------
    # (b) Optimal LR vs batch size
    # ------------------------------------------------------------------
    lr_data = _collect_lr_sweep(lr_dir)
    for opt_key, display in [("muon", "Muon"), ("adamw", "AdamW")]:
        if opt_key not in lr_data:
            continue
        bs_dict = lr_data[opt_key]
        batch_sizes = sorted(bs_dict.keys())
        best_lrs = []
        for bs in batch_sizes:
            pairs = bs_dict[bs]  # list of (lr, final_train_loss)
            # Find LR with minimum final train loss
            best_pair = min(pairs, key=lambda x: x[1])
            best_lrs.append(best_pair[0])

        ax_b.plot(batch_sizes, best_lrs, color=get_color(display),
                  marker=get_marker(display), label=display,
                  markeredgecolor="black", markeredgewidth=0.3)

    ax_b.set_xscale("log", base=2)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Batch size")
    ax_b.set_ylabel("Optimal learning rate")
    ax_b.legend(loc="upper left")
    add_panel_label(ax_b, "b")

    save_fig(fig, "fig6_nanogpt_bcrit", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 6 -- NanoGPT B_crit")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
