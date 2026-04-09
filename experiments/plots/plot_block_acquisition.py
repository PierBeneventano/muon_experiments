#!/usr/bin/env python3
"""
Figure 5 — Block acquisition trajectories (K=4).

Muon acquires all blocks concurrently; GD acquires sequentially or fails.

Reads: results/matrix_sensing/05_block_acquisition.json
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


def load_data(results_dir: str) -> dict:
    path = Path(results_dir) / "matrix_sensing" / "05_block_acquisition.json"
    with open(path) as f:
        return json.load(f)


def plot(results_dir: str, output_dir: str):
    setup_style()
    data = load_data(results_dir)

    n_blocks = data["config"]["K"]
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL, sharey=True)

    for ax, opt in zip(axes, ["Muon", "GD"]):
        # data["detailed"][opt] is a list of per-seed dicts.
        # Each has "trajectory": list of {step, block_masses} dicts.
        # Average block_masses across seeds at each time step.
        seed_runs = data["detailed"][opt]

        # Use first seed to determine steps and n_steps
        traj_0 = seed_runs[0]["trajectory"]
        steps = np.array([t["step"] for t in traj_0])
        n_traj = len(traj_0)

        # Collect block masses: shape (n_seeds, n_timesteps, K)
        all_masses = []
        for run in seed_runs:
            traj = run["trajectory"]
            masses = np.array([t["block_masses"] for t in traj[:n_traj]])
            all_masses.append(masses)
        all_masses = np.array(all_masses)  # (n_seeds, n_timesteps, K)

        # Mean across seeds
        mean_masses = np.mean(all_masses, axis=0)  # (n_timesteps, K)

        for k in range(n_blocks):
            shade = 0.3 + 0.7 * k / max(n_blocks - 1, 1)
            ax.plot(steps, mean_masses[:, k], color=get_color(opt),
                    alpha=shade, label=f"Block {k+1}")

        ax.set_xlabel("Training step")
        ax.set_title(opt, fontsize=9)
        ax.legend(loc="lower right", fontsize=6, ncol=2)

    axes[0].set_ylabel("Block mass")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")

    save_fig(fig, "fig5_block_acquisition", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 5 — Block acquisition")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
