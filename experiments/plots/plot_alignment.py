#!/usr/bin/env python3
"""
Figure 4 — Alignment dynamics.

(a) Principal angle between W and G singular frames over training, with
    3-phase regions shaded (exploration / alignment / convergence).
(b) Histogram of per-step advantage (dH_polar - dH_gd), showing asymmetry.

Reads: results/matrix_sensing/08_alignment_tracking.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from plots.style import setup_style, get_color, save_fig, add_panel_label, DOUBLE_COL
except ImportError:
    from style import setup_style, get_color, save_fig, add_panel_label, DOUBLE_COL


PHASE_COLORS = ["#ffe0b2", "#c8e6c9", "#bbdefb"]  # warm, green, cool
PHASE_LABELS = ["Exploration", "Alignment", "Convergence"]


def load_data(results_dir: str) -> dict:
    path = Path(results_dir) / "matrix_sensing" / "08_alignment_tracking.json"
    with open(path) as f:
        return json.load(f)


def plot(results_dir: str, output_dir: str):
    setup_style()
    data = load_data(results_dir)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # ------------------------------------------------------------------
    # (a) Mean alignment cosine over training (averaged across seeds)
    # ------------------------------------------------------------------
    # data["aggregated_alignment"] is a list of dicts with keys:
    #   step, mean_cos_left, mean_cos_right, mean_H
    agg = data["aggregated_alignment"]
    steps = np.array([t["step"] for t in agg])
    cos_left = np.array([t["mean_cos_left"] for t in agg])
    cos_right = np.array([t["mean_cos_right"] for t in agg])

    ax_a.plot(steps, cos_left, color=get_color("Muon"),
              label="Left singular vectors")
    ax_a.plot(steps, cos_right, color=get_color("GD"),
              label="Right singular vectors")

    ax_a.set_xlabel("Training step")
    ax_a.set_ylabel("Mean cosine similarity\n(W vs G frames)")
    ax_a.legend(loc="upper right", fontsize=6)
    add_panel_label(ax_a, "a")

    # ------------------------------------------------------------------
    # (b) Per-step dH histogram across all seeds
    # ------------------------------------------------------------------
    # Collect all per-step dH values from per_seed tracking
    all_dH = []
    for seed_data in data["per_seed"]:
        for t in seed_data["tracking"]:
            if t["dH"] != 0.0:  # skip initial step with dH=0
                all_dH.append(t["dH"])
    all_dH = np.array(all_dH)

    ax_b.hist(all_dH, bins=40, color=get_color("Muon"),
              edgecolor="black", linewidth=0.3, alpha=0.8)
    ax_b.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax_b.axvline(np.mean(all_dH), color=get_color("Muon"),
                 linewidth=1.0, linestyle="-",
                 label=f"mean = {np.mean(all_dH):.4f}")

    ax_b.set_xlabel(r"$\Delta H$ per step")
    ax_b.set_ylabel("Count")
    ax_b.legend(loc="upper left")
    add_panel_label(ax_b, "b")

    save_fig(fig, "fig4_alignment", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 4 — Alignment dynamics")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
