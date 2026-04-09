#!/usr/bin/env python3
"""
Figure 3 — Condition-number scaling.

(a) Advantage = H_muon - H_gd vs log(kappa) with linear regression line and R^2.
(b) Two-factor decomposition bar chart: isometry vs alignment vs magnitude.

Reads:
    results/matrix_sensing/03_kappa_scaling.json
    results/matrix_sensing/07_factorial_2x2.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

try:
    from plots.style import setup_style, get_color, save_fig, add_panel_label, DOUBLE_COL
except ImportError:
    from style import setup_style, get_color, save_fig, add_panel_label, DOUBLE_COL


def load_data(results_dir: str):
    base = Path(results_dir) / "matrix_sensing"
    with open(base / "03_kappa_scaling.json") as f:
        kappa_data = json.load(f)
    with open(base / "07_factorial_2x2.json") as f:
        factorial_data = json.load(f)
    return kappa_data, factorial_data


def plot(results_dir: str, output_dir: str):
    setup_style()
    kappa_data, factorial_data = load_data(results_dir)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # ------------------------------------------------------------------
    # (a) Advantage vs log(kappa)
    # ------------------------------------------------------------------
    # Data is in kappa_data["per_kappa"] keyed by kappa string values
    per_kappa = kappa_data["per_kappa"]
    kappas_str = sorted(per_kappa.keys(), key=float)
    kappas = np.array([float(k) for k in kappas_str])
    log_kappa = np.log10(kappas)
    advantage_mean = np.array([per_kappa[k]["H_advantage_mean"] for k in kappas_str])
    advantage_std = np.array([per_kappa[k]["H_advantage_std"] for k in kappas_str])

    ax_a.errorbar(log_kappa, advantage_mean, yerr=advantage_std,
                  fmt="o", ms=5, color=get_color("Muon"),
                  ecolor=get_color("Muon"), elinewidth=0.8, capsize=3,
                  markeredgecolor="black", markeredgewidth=0.3, zorder=5)

    # Linear regression (use pre-computed if available, else compute)
    if "scaling_analysis" in kappa_data:
        sa = kappa_data["scaling_analysis"]
        slope = sa["regression_slope"]
        intercept = sa["regression_intercept"]
        r2 = sa["regression_R2"]
    else:
        slope, intercept, r, p, se = sp_stats.linregress(log_kappa, advantage_mean)
        r2 = r ** 2

    x_fit = np.linspace(log_kappa.min(), log_kappa.max(), 100)
    ax_a.plot(x_fit, slope * x_fit + intercept, "--", color="black",
              linewidth=0.9, label=f"$R^2 = {r2:.2f}$")

    ax_a.set_xlabel(r"$\log_{10}(\kappa)$")
    ax_a.set_ylabel(r"Advantage  $H_{\mathrm{Muon}} - H_{\mathrm{GD}}$")
    ax_a.legend(loc="upper left")
    add_panel_label(ax_a, "a")

    # ------------------------------------------------------------------
    # (b) Two-factor decomposition bar chart
    # ------------------------------------------------------------------
    # factorial_data["factorial_decomposition"]["H"] has polar_main_effect,
    # norm_main_effect, interaction
    decomp = factorial_data["factorial_decomposition"]["H"]
    factors = ["Polar (isometry)", "Norm", "Interaction"]
    factor_keys = ["polar_main_effect", "norm_main_effect", "interaction"]
    values = [decomp[k] for k in factor_keys]
    bar_colors = ["#0072B2", "#009E73", "#E69F00"]

    x = np.arange(len(factors))
    ax_b.bar(x, values, color=bar_colors,
             edgecolor="black", linewidth=0.4)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(factors, rotation=15, ha="right")
    ax_b.set_ylabel("Effect size on $\\Delta H$")
    ax_b.axhline(0, color="black", linewidth=0.4)
    add_panel_label(ax_b, "b")

    save_fig(fig, "fig3_kappa_scaling", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Figure 3 — Kappa scaling")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
