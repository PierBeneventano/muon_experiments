#!/usr/bin/env python3
"""
Figure 8 — 1500-configuration ablation heatmap.

Heatmap of H_muon - H_gd across 1500 configurations.
Rows: (m, p) pairs.  Columns: init_scale, rank.

Reads: results/matrix_sensing/04_1500_config_ablation.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from plots.style import setup_style, save_fig, add_panel_label, DOUBLE_COL_TALL
except ImportError:
    from style import setup_style, save_fig, add_panel_label, DOUBLE_COL_TALL


def load_data(results_dir: str) -> dict:
    path = Path(results_dir) / "matrix_sensing" / "04_1500_config_ablation.json"
    with open(path) as f:
        return json.load(f)


def plot(results_dir: str, output_dir: str):
    setup_style()
    data = load_data(results_dir)

    fig, ax = plt.subplots(1, 1, figsize=DOUBLE_COL_TALL)

    # Build the 2-D advantage matrix from data["per_config"].
    # Each entry has config = {m, n, p, init_scale, rank} and H_advantage.
    # Rows: (m, p) pairs.  Columns: (init_scale, rank) pairs.
    per_config = data["per_config"]

    # Discover unique row/col values
    row_keys = sorted({(c["config"]["m"], c["config"]["p"]) for c in per_config})
    col_keys = sorted({(c["config"]["init_scale"], c["config"]["rank"]) for c in per_config})

    row_idx = {k: i for i, k in enumerate(row_keys)}
    col_idx = {k: i for i, k in enumerate(col_keys)}

    mat = np.full((len(row_keys), len(col_keys)), np.nan)
    for c in per_config:
        cfg = c["config"]
        ri = row_idx[(cfg["m"], cfg["p"])]
        ci = col_idx[(cfg["init_scale"], cfg["rank"])]
        mat[ri, ci] = c["H_advantage"]

    row_labels = [f"m={m}, p={p}" for m, p in row_keys]
    col_labels = [f"s={s}, r={r}" for s, r in col_keys]

    # Diverging colourmap centred at 0
    vmax = np.nanmax(np.abs(mat))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$H_{\mathrm{Muon}} - H_{\mathrm{GD}}$", fontsize=8)

    # Tick labels
    n_rows, n_cols = mat.shape
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha="right")

    ax.set_xlabel("(init_scale, rank)")
    ax.set_ylabel("(m, p)")
    n_configs = data.get("total_configs", len(per_config))
    ax.set_title(f"Entropy advantage across {n_configs} configurations", fontsize=9)

    save_fig(fig, "fig8_1500_ablation", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 8 — 1500-config ablation")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
