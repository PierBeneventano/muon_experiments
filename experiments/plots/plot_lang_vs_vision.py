#!/usr/bin/env python3
"""
Figure 6 -- Language vs vision rank collapse.

Two-modality comparison: AdamW collapses effective rank on language pretraining
but not on vision; Muon preserves rank on both. This figure is the empirical
core of the paper's H3 argument in Section 7 ("Why Language Specifically").

Reads:
  Language:  results/nanogpt/04_spectral_tracking/{muon,adamw}_s{42,137,2024}/
             spectral/spectral_log.jsonl
  Vision:    results/vision/lang_vs_vision/vit_{muon,adamw}_s{42,137,2024}/
             spectral/spectral_log.jsonl

Produces:
  results/plots/fig6_lang_vs_vision.pdf + .png

Layout: 2x2 grid
  (a) Language, stable rank      (b) Vision, stable rank
  (c) Language, spectral entropy (d) Vision, spectral entropy

Each curve = mean across 3 seeds; shading = ±1 std.

Usage:
  python experiments/plots/plot_lang_vs_vision.py
  python experiments/plots/plot_lang_vs_vision.py --results_dir path/to/results
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from plots.style import (
        setup_style, get_color, get_marker, save_fig, add_panel_label,
        DOUBLE_COL_TALL,
    )
except ImportError:
    from style import (
        setup_style, get_color, get_marker, save_fig, add_panel_label,
        DOUBLE_COL_TALL,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_spectral_log(path: Path):
    """Load a spectral_log.jsonl. Returns list of dicts with 'iter' and
    per-layer 'spectral_entropy', 'stable_rank' (if present)."""
    snapshots = []
    if not path.is_file():
        return snapshots
    with open(path) as f:
        for line in f:
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return snapshots


def _avg_over_layers(snapshot, metric_key: str):
    """Average `metric_key` across all layer records with it defined.
    Returns NaN if no record has it."""
    vals = []
    for rec in snapshot.get("layers", []):
        if metric_key in rec and rec[metric_key] is not None:
            vals.append(float(rec[metric_key]))
    return float(np.mean(vals)) if vals else float("nan")


def _collect_modality(run_dir_glob: Path, optimizers=("muon", "adamw"),
                      seeds=(42, 137, 2024)):
    """For a given run directory (language or vision), return:
        { optimizer: { 'iters': [...], 'stable_rank_per_seed': [[...], ...],
                       'spectral_entropy_per_seed': [[...], ...] } }
    """
    out = {}
    for opt in optimizers:
        opt_data = {
            "iters": None,
            "stable_rank": [],     # list-of-lists: one list per seed
            "spectral_entropy": [],
        }
        found = 0
        for seed in seeds:
            # Language dir: {opt}_s{seed}; vision dir: vit_{opt}_s{seed}
            # Try both naming patterns
            candidates = [
                run_dir_glob.parent / f"{opt}_s{seed}" / "spectral" / "spectral_log.jsonl",
                run_dir_glob.parent / f"vit_{opt}_s{seed}" / "spectral" / "spectral_log.jsonl",
            ]
            log_path = None
            for c in candidates:
                if c.is_file():
                    log_path = c
                    break
            if log_path is None:
                warnings.warn(f"No spectral log for {opt} seed {seed} under {run_dir_glob.parent}")
                continue
            snapshots = _load_spectral_log(log_path)
            if not snapshots:
                continue
            found += 1
            iters = [s["iter"] for s in snapshots]
            sr = [_avg_over_layers(s, "stable_rank") for s in snapshots]
            se = [_avg_over_layers(s, "spectral_entropy") for s in snapshots]
            if opt_data["iters"] is None:
                opt_data["iters"] = iters
            opt_data["stable_rank"].append(sr)
            opt_data["spectral_entropy"].append(se)

        if found > 0:
            out[opt] = opt_data
    return out


def _reduce(seed_lists):
    """Given a list of per-seed curves, return (mean, std, n) aligned to the
    shortest curve length so missing tails don't drag the mean."""
    if not seed_lists:
        return None, None, 0
    min_len = min(len(s) for s in seed_lists)
    arr = np.array([s[:min_len] for s in seed_lists], dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0) if len(seed_lists) > 1 else np.zeros_like(mean)
    return mean, std, len(seed_lists)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel(ax, data, metric_key: str, ylabel: str, title: str,
                ylog=False):
    """Plot one modality × metric panel."""
    if not data:
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", va="center", color="#888888", fontsize=10)
        ax.set_title(title)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        return

    for opt in ("muon", "adamw"):
        if opt not in data:
            continue
        d = data[opt]
        iters = d["iters"]
        if iters is None:
            continue
        series = d[metric_key]
        mean, std, n = _reduce(series)
        if mean is None:
            continue
        iters_clip = iters[:len(mean)]
        display = "Muon" if opt == "muon" else "AdamW"
        color = get_color(display)
        ax.plot(iters_clip, mean, label=f"{display} (n={n})",
                color=color, marker=get_marker(display), markersize=3,
                markeredgecolor="black", markeredgewidth=0.3, markevery=max(1, len(iters_clip)//10))
        ax.fill_between(iters_clip, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylog:
        ax.set_yscale("log")
    ax.legend(loc="best")


def plot(results_dir: str, output_dir: str):
    setup_style()

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)

    # Point `_collect_modality` at an arbitrary child directory of the sweep
    # (its `parent` is the sweep root).
    lang_root = results_dir / "nanogpt" / "04_spectral_tracking" / "_any"
    vision_root = results_dir / "vision" / "lang_vs_vision" / "_any"

    print(f"[fig6] Language from: {lang_root.parent}")
    print(f"[fig6] Vision from:   {vision_root.parent}")

    lang = _collect_modality(lang_root)
    vision = _collect_modality(vision_root)

    print(f"[fig6] Language optimizers found: {sorted(lang.keys())}")
    print(f"[fig6] Vision optimizers found:   {sorted(vision.keys())}")

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_TALL[0], DOUBLE_COL_TALL[1] + 1.5))

    _plot_panel(axes[0, 0], lang,   "stable_rank",      "Stable rank",       "Language (Shakespeare)")
    _plot_panel(axes[0, 1], vision, "stable_rank",      "Stable rank",       "Vision (CIFAR-10)")
    _plot_panel(axes[1, 0], lang,   "spectral_entropy", "Spectral entropy",  "Language (Shakespeare)")
    _plot_panel(axes[1, 1], vision, "spectral_entropy", "Spectral entropy",  "Vision (CIFAR-10)")

    for ax, label in zip(axes.flat, "abcd"):
        add_panel_label(ax, label)

    fig.suptitle("AdamW collapses rank on language but not on vision; Muon preserves rank on both",
                 fontsize=10, y=1.02)

    save_fig(fig, "fig6_lang_vs_vision", str(output_dir.parent), formats=("pdf", "png"))


def main():
    p = argparse.ArgumentParser(description="Figure 6 -- language vs vision rank collapse")
    p.add_argument("--results_dir", type=str,
                   default=None,
                   help="Path to project's results/ directory")
    p.add_argument("--output_dir", type=str,
                   default=None,
                   help="Output directory (defaults to results_dir/plots)")
    args = p.parse_args()

    # Sensible defaults anchored at project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else project_root / "experiments" / "results"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"

    plot(str(results_dir), str(output_dir))


if __name__ == "__main__":
    main()
