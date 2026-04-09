#!/usr/bin/env python3
"""
Figure 7 — NanoGPT spectral entropy.

(a) Mean spectral entropy across all weight matrices during training: Muon vs AdamW.
(b) Per-layer spectral entropy comparison at convergence (grouped bar chart).

Reads: results/nanogpt/04_spectral_tracking/
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


def _load_spectral(spectral_dir: str) -> dict:
    """
    Expects files named <optimizer>_spectral.json with structure:
      { "optimizer": "...",
        "global_entropy": {"steps": [...], "mean": [...], "std": [...]},
        "per_layer_entropy": {"layer_names": [...], "values": [...]} }
    """
    data = {}
    for p in sorted(Path(spectral_dir).glob("*_spectral.json")):
        with open(p) as f:
            d = json.load(f)
        data[d["optimizer"]] = d
    return data


def plot(results_dir: str, output_dir: str):
    setup_style()
    spectral_dir = Path(results_dir) / "nanogpt" / "04_spectral_tracking"
    data = _load_spectral(str(spectral_dir))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    opt_pairs = [("muon", "Muon"), ("adamw", "AdamW")]

    # ------------------------------------------------------------------
    # (a) Global entropy over training
    # ------------------------------------------------------------------
    for key, display in opt_pairs:
        if key not in data:
            continue
        g = data[key]["global_entropy"]
        steps = np.array(g["steps"])
        mean = np.array(g["mean"])
        std = np.array(g.get("std", np.zeros_like(mean)))
        ax_a.plot(steps, mean, color=get_color(display), label=display)
        ax_a.fill_between(steps, mean - std, mean + std,
                          color=get_color(display), alpha=0.15)

    ax_a.set_xlabel("Training step")
    ax_a.set_ylabel("Mean spectral entropy $H$")
    ax_a.legend(loc="lower right")
    add_panel_label(ax_a, "a")

    # ------------------------------------------------------------------
    # (b) Per-layer entropy at convergence (grouped bar)
    # ------------------------------------------------------------------
    # Use the first available optimizer to get layer names
    ref_key = next((k for k, _ in opt_pairs if k in data), None)
    if ref_key is None:
        save_fig(fig, "fig7_nanogpt_spectral", output_dir)
        return

    layer_names = data[ref_key]["per_layer_entropy"]["layer_names"]
    n_layers = len(layer_names)
    x = np.arange(n_layers)
    width = 0.35

    for i, (key, display) in enumerate(opt_pairs):
        if key not in data:
            continue
        vals = np.array(data[key]["per_layer_entropy"]["values"])
        offset = (i - 0.5) * width
        ax_b.bar(x + offset, vals, width, color=get_color(display),
                 edgecolor="black", linewidth=0.3, label=display)

    ax_b.set_xticks(x)
    short_names = [n.replace("transformer.", "").replace(".weight", "")
                   for n in layer_names]
    ax_b.set_xticklabels(short_names, rotation=55, ha="right", fontsize=5)
    ax_b.set_ylabel("Spectral entropy $H$")
    ax_b.legend(loc="upper right")
    add_panel_label(ax_b, "b")

    save_fig(fig, "fig7_nanogpt_spectral", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 7 — NanoGPT spectral")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
