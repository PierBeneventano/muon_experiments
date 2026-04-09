#!/usr/bin/env python3
"""
Table 1 — Normalizer taxonomy (rendered as a figure).

Optimizer name | update form | sublinearity index rho | diagonal ATSR.
Hardcoded — no data file needed.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib

try:
    from plots.style import setup_style, save_fig, SINGLE_COL
except ImportError:
    from style import setup_style, save_fig, SINGLE_COL


# ---- Taxonomy data (hardcoded) ------------------------------------------
ROWS = [
    # (Optimizer, Update form, rho, Diag ATSR)
    ("GD",          r"$-\eta G$",                                "1",   "No"),
    ("NM-GD",       r"$-\eta G / \|G\|_F$",                     "0",   "No"),
    ("Muon",        r"$-\eta U V^\top$  (polar of $G$)",        "0",   "Yes"),
    ("Adam",        r"$-\eta \hat{m} / (\sqrt{\hat{v}}+\varepsilon)$", "0.5", "Partial"),
    ("LARS",        r"$-\eta \|W\| G / \|G\|$",                 "0",   "No"),
    ("Random-Orth", r"$-\eta Q$  ($Q$ Haar-random)",            "0",   "Yes"),
]

HEADERS = ["Optimizer", "Update form", r"$\rho$", "Diag ATSR"]


def plot(results_dir: str, output_dir: str):
    setup_style()

    fig_height = 0.35 * (len(ROWS) + 1) + 0.4
    fig, ax = plt.subplots(figsize=(SINGLE_COL[0] * 2, fig_height))
    ax.axis("off")

    cell_text = [[r[0], r[1], r[2], r[3]] for r in ROWS]

    table = ax.table(
        cellText=cell_text,
        colLabels=HEADERS,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(len(HEADERS)):
        cell = table[(0, j)]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#e0e0e0")
        cell.set_edgecolor("black")

    # Alternate row shading
    for i in range(1, len(ROWS) + 1):
        for j in range(len(HEADERS)):
            cell = table[(i, j)]
            cell.set_edgecolor("#999999")
            if i % 2 == 0:
                cell.set_facecolor("#f5f5f5")
            else:
                cell.set_facecolor("white")

    ax.set_title("Table 1: Normalizer taxonomy", fontsize=9, pad=12)

    save_fig(fig, "table1_taxonomy", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Table 1 — Taxonomy")
    parser.add_argument("--results_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
