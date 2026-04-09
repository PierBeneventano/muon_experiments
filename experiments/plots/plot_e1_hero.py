#!/usr/bin/env python3
"""
Figure 1 — Hero figure (three panels).

(a) Singular-value spectra at convergence: Muon (near-uniform) vs GD (peaked) vs NM-GD.
(b) Bar chart: mean spectral entropy +/- std for all 4 optimizers, with significance stars.
(c) Spectral entropy trajectories over training steps.

Reads: results/matrix_sensing/01_e1_four_way.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running as ``python -m plots.plot_e1_hero`` or standalone
try:
    from plots.style import setup_style, get_color, get_marker, save_fig, add_panel_label, DOUBLE_COL_TALL
except ImportError:
    from style import setup_style, get_color, get_marker, save_fig, add_panel_label, DOUBLE_COL_TALL


def load_data(results_dir: str) -> dict:
    path = Path(results_dir) / "matrix_sensing" / "01_e1_four_way.json"
    with open(path) as f:
        return json.load(f)


def plot(results_dir: str, output_dir: str):
    setup_style()
    data = load_data(results_dir)

    optimizers = ["Muon", "GD", "NM-GD", "Random-Orth"]

    # Data lives in data["H_values_per_optimizer"][opt] (list of H per seed)
    # and data["summary"][opt] with H_mean / H_std.
    # Pairwise tests in data["pairwise_tests"]["Muon_vs_GD"] etc.
    # No spectra or trajectories stored, so we make a 2-panel figure:
    #   (a) strip/box plot of per-seed H distributions
    #   (b) bar chart of mean H +/- std with significance stars

    fig, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=DOUBLE_COL_TALL)

    # ------------------------------------------------------------------
    # (a) Per-seed H distributions (strip + box)
    # ------------------------------------------------------------------
    all_vals = []
    for opt in optimizers:
        vals = np.array(data["H_values_per_optimizer"][opt])
        all_vals.append(vals)

    bp = ax_box.boxplot(all_vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.0))
    for patch, opt in zip(bp["boxes"], optimizers):
        patch.set_facecolor(get_color(opt))
        patch.set_alpha(0.6)
    # Overlay individual points
    for i, (opt, vals) in enumerate(zip(optimizers, all_vals)):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax_box.scatter(np.full_like(vals, i + 1) + jitter, vals,
                       s=12, color=get_color(opt), edgecolor="black",
                       linewidth=0.3, zorder=5)

    ax_box.set_xticklabels(optimizers, rotation=25, ha="right")
    ax_box.set_ylabel("Spectral entropy $H$")
    add_panel_label(ax_box, "a")

    # ------------------------------------------------------------------
    # (b) Mean spectral entropy bar chart with significance stars
    # ------------------------------------------------------------------
    means, stds, colors = [], [], []
    for opt in optimizers:
        vals = np.array(data["H_values_per_optimizer"][opt])
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        colors.append(get_color(opt))

    x = np.arange(len(optimizers))
    ax_bar.bar(x, means, yerr=stds, color=colors,
               edgecolor="black", linewidth=0.4,
               capsize=3, error_kw={"linewidth": 0.8})
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(optimizers, rotation=25, ha="right")
    ax_bar.set_ylabel("Spectral entropy $H$")

    # Significance stars from pairwise_tests (Bonferroni-corrected)
    if "pairwise_tests" in data:
        max_y = max(m + s for m, s in zip(means, stds)) * 1.05
        for i, opt in enumerate(optimizers[1:], 1):
            pair_key = f"Muon_vs_{opt}"
            if pair_key not in data["pairwise_tests"]:
                continue
            p = data["pairwise_tests"][pair_key]["p_value_bonferroni"]
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                continue
            y = max_y + 0.03 * i
            ax_bar.plot([0, i], [y, y], color="black", linewidth=0.6)
            ax_bar.text((0 + i) / 2, y + 0.005, stars,
                        ha="center", va="bottom", fontsize=7)

    add_panel_label(ax_bar, "b")

    save_fig(fig, "fig1_hero", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 1 — Hero figure")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
