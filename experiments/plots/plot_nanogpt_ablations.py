#!/usr/bin/env python3
"""
Figure 9 -- NanoGPT ablation experiments.

(a) Weight decay ablation  -- val loss vs weight decay for Muon and AdamW.
(b) Momentum ablation      -- val loss vs beta for Muon.
(c) Model scale            -- val loss vs n_embd for Muon and AdamW.
(d) Head count ablation    -- val loss vs n_heads for Muon and AdamW.

Reads:
    results/nanogpt/06_weight_decay_ablation/*/summary.json
    results/nanogpt/07_momentum_ablation/*/summary.json
    results/nanogpt/08_model_scale/*/summary.json
    results/nanogpt/10_head_ablation/*/summary.json

Each summary.json is expected to contain at minimum:
    {
        "optimizer": "muon" | "adamw",
        "best_val_loss": float,
        "final_train_loss": float,
        "config": { ... }
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict

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
# Helpers
# ---------------------------------------------------------------------------

def _get_val_loss(summary: dict) -> float | None:
    """Extract the best validation loss from a summary dict."""
    if "best_val_loss" in summary and summary["best_val_loss"] is not None:
        return float(summary["best_val_loss"])
    if "final_val_loss" in summary and summary["final_val_loss"] is not None:
        return float(summary["final_val_loss"])
    if "val_losses" in summary and isinstance(summary["val_losses"], list):
        vals = summary["val_losses"]
        if vals:
            v = vals[-1]
            return float(v["loss"] if isinstance(v, dict) else v)
    return None


def _collect_runs(sweep_dir: Path) -> list[dict]:
    """Load all summary.json files under *sweep_dir* (one level deep)."""
    runs = []
    if not sweep_dir.is_dir():
        return runs
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            d = json.load(f)
        d["_run_dir"] = str(run_dir)
        d["_dir_name"] = run_dir.name
        runs.append(d)
    return runs


# ---------------------------------------------------------------------------
# Per-experiment collectors
# ---------------------------------------------------------------------------

def _collect_weight_decay(sweep_dir: Path) -> dict:
    """
    Returns {optimizer: {weight_decay: [val_losses]}}.

    Parses weight decay from directory name pattern: {opt}_wd{wd}_s{seed}
    or from summary.json config.
    """
    results = defaultdict(lambda: defaultdict(list))
    wd_pattern = re.compile(r"^(muon|adamw)_wd([\d.]+)_s\d+$")

    for run in _collect_runs(sweep_dir):
        val_loss = _get_val_loss(run)
        if val_loss is None:
            continue

        opt = run.get("optimizer", "").lower()

        # Try config first, fall back to directory name parsing
        wd = None
        cfg = run.get("config", {})
        if "weight_decay" in cfg:
            wd = float(cfg["weight_decay"])
        elif "weight_decay" in run:
            wd = float(run["weight_decay"])
        else:
            m = wd_pattern.match(run.get("_dir_name", ""))
            if m:
                opt = opt or m.group(1)
                wd = float(m.group(2))

        if opt and wd is not None:
            display = "Muon" if opt == "muon" else "AdamW"
            results[display][wd].append(val_loss)

    return results


def _collect_momentum(sweep_dir: Path) -> dict:
    """
    Returns {"Muon": {beta: [val_losses]}}.

    Parses momentum from directory name pattern: mom{beta}_s{seed}
    or from summary.json config.
    """
    results = defaultdict(lambda: defaultdict(list))
    mom_pattern = re.compile(r"^mom([\d.]+)_s\d+$")

    for run in _collect_runs(sweep_dir):
        val_loss = _get_val_loss(run)
        if val_loss is None:
            continue

        beta = None
        cfg = run.get("config", {})
        for key in ("momentum", "beta", "beta1"):
            if key in cfg:
                beta = float(cfg[key])
                break
        if beta is None and "momentum" in run:
            beta = float(run["momentum"])

        if beta is None:
            m = mom_pattern.match(run.get("_dir_name", ""))
            if m:
                beta = float(m.group(1))

        if beta is not None:
            results["Muon"][beta].append(val_loss)

    return results


def _collect_model_scale(sweep_dir: Path) -> dict:
    """
    Returns {optimizer: {n_embd: [val_losses]}}.

    Parses n_embd from directory name pattern: {opt}_embd{dim}_s{seed}
    or from summary.json config.
    """
    results = defaultdict(lambda: defaultdict(list))
    embd_pattern = re.compile(r"^(muon|adamw)_embd(\d+)_s\d+$")

    for run in _collect_runs(sweep_dir):
        val_loss = _get_val_loss(run)
        if val_loss is None:
            continue

        opt = run.get("optimizer", "").lower()
        n_embd = None

        cfg = run.get("config", {})
        if "n_embd" in cfg:
            n_embd = int(cfg["n_embd"])
        elif "n_embd" in run:
            n_embd = int(run["n_embd"])
        else:
            m = embd_pattern.match(run.get("_dir_name", ""))
            if m:
                opt = opt or m.group(1)
                n_embd = int(m.group(2))

        if opt and n_embd is not None:
            display = "Muon" if opt == "muon" else "AdamW"
            results[display][n_embd].append(val_loss)

    return results


def _collect_head_ablation(sweep_dir: Path) -> dict:
    """
    Returns {optimizer: {n_heads: [val_losses]}}.

    Parses n_heads from directory name pattern: {opt}_heads{n}_s{seed}
    or from summary.json config.
    """
    results = defaultdict(lambda: defaultdict(list))
    heads_pattern = re.compile(r"^(muon|adamw)_heads(\d+)_s\d+$")

    for run in _collect_runs(sweep_dir):
        val_loss = _get_val_loss(run)
        if val_loss is None:
            continue

        opt = run.get("optimizer", "").lower()
        n_heads = None

        cfg = run.get("config", {})
        if "n_head" in cfg:
            n_heads = int(cfg["n_head"])
        elif "n_heads" in cfg:
            n_heads = int(cfg["n_heads"])
        elif "n_head" in run:
            n_heads = int(run["n_head"])
        elif "n_heads" in run:
            n_heads = int(run["n_heads"])
        else:
            m = heads_pattern.match(run.get("_dir_name", ""))
            if m:
                opt = opt or m.group(1)
                n_heads = int(m.group(2))

        if opt and n_heads is not None:
            display = "Muon" if opt == "muon" else "AdamW"
            results[display][n_heads].append(val_loss)

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_optimizer_curves(ax, data: dict, xlabel: str, xscale: str = "linear",
                           log_x: bool = False):
    """
    Plot mean +/- std for each optimizer in *data*.

    data: {display_name: {x_value: [val_losses]}}
    """
    for display in ["Muon", "AdamW"]:
        if display not in data:
            continue
        xvals = sorted(data[display].keys())
        means, stds = [], []
        for x in xvals:
            arr = np.array(data[display][x])
            means.append(np.mean(arr))
            stds.append(np.std(arr))
        means, stds = np.array(means), np.array(stds)

        ax.errorbar(xvals, means, yerr=stds,
                    color=get_color(display), marker=get_marker(display),
                    label=display, capsize=3, linewidth=1.0,
                    markeredgecolor="black", markeredgewidth=0.3)

    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best validation loss")
    ax.legend(loc="best")


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot(results_dir: str, output_dir: str):
    """Generate figure 9: NanoGPT ablations (2x2 grid)."""
    setup_style()

    base = Path(results_dir) / "nanogpt"

    # Collect data for each panel
    panels = {
        "a": {
            "title": "Weight decay",
            "dir": base / "06_weight_decay_ablation",
            "collector": _collect_weight_decay,
            "xlabel": "Weight decay",
            "log_x": False,
        },
        "b": {
            "title": "Momentum",
            "dir": base / "07_momentum_ablation",
            "collector": _collect_momentum,
            "xlabel": r"Momentum ($\beta$)",
            "log_x": False,
        },
        "c": {
            "title": "Model scale",
            "dir": base / "08_model_scale",
            "collector": _collect_model_scale,
            "xlabel": r"$n_\mathrm{embd}$",
            "log_x": True,
        },
        "d": {
            "title": "Head count",
            "dir": base / "10_head_ablation",
            "collector": _collect_head_ablation,
            "xlabel": r"$n_\mathrm{heads}$",
            "log_x": True,
        },
    }

    # 2x2 figure — use DOUBLE_COL_TALL width with taller height for 2 rows
    fig_width = DOUBLE_COL_TALL[0]
    fig_height = DOUBLE_COL_TALL[1] * 2  # two rows
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    ax_map = {
        "a": axes[0, 0],
        "b": axes[0, 1],
        "c": axes[1, 0],
        "d": axes[1, 1],
    }

    any_data = False

    for label in ("a", "b", "c", "d"):
        ax = ax_map[label]
        p = panels[label]

        if not p["dir"].is_dir():
            warnings.warn(f"Panel ({label}): directory not found: {p['dir']}")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            ax.set_title(p["title"])
            add_panel_label(ax, label)
            continue

        data = p["collector"](p["dir"])

        if not data or all(len(v) == 0 for v in data.values()):
            warnings.warn(
                f"Panel ({label}): no summary.json files with val loss in {p['dir']}"
            )
            ax.text(0.5, 0.5, "No data yet\n(waiting for runs)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="gray")
            ax.set_title(p["title"])
            add_panel_label(ax, label)
            continue

        any_data = True
        _plot_optimizer_curves(ax, data, xlabel=p["xlabel"], log_x=p["log_x"])
        ax.set_title(p["title"])
        add_panel_label(ax, label)

    if not any_data:
        print("  WARNING: No data found for any ablation panel. "
              "Figure saved with placeholder text.")

    save_fig(fig, "fig9_nanogpt_ablations", output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Figure 9 -- NanoGPT ablation experiments"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Root results directory (contains nanogpt/ subdirectory). "
             "Defaults to experiments/results relative to project root.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root output directory (plots saved to <output_dir>/plots/). "
             "Defaults to same as results_dir.",
    )
    args = parser.parse_args()

    # Resolve defaults relative to project root
    project_root = Path(__file__).resolve().parent.parent
    results_dir = args.results_dir or str(project_root / "results")
    output_dir = args.output_dir or results_dir

    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}")

    plot(results_dir, output_dir)


if __name__ == "__main__":
    main()
