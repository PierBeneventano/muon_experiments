"""
Consistent NeurIPS-quality matplotlib style for Muon implicit-bias paper.

Usage:
    from plots.style import setup_style, get_color, get_marker, save_fig
    setup_style()
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011 + custom tweaks)
# ---------------------------------------------------------------------------
COLORS = {
    "Muon":        "#0072B2",  # blue
    "GD":          "#E69F00",  # orange
    "NM-GD":       "#D55E00",  # red-orange
    "Random-Orth":  "#009E73",  # green
    "Adam":        "#CC79A7",  # purple-pink
    "AdamW":       "#CC79A7",  # same family as Adam
    "LARS":        "#8C564B",  # brown
    "SGD":         "#E69F00",  # alias for GD
}

MARKERS = {
    "Muon":        "o",
    "GD":          "s",
    "NM-GD":       "^",
    "Random-Orth":  "D",
    "Adam":        "v",
    "AdamW":       "v",
    "LARS":        "P",
    "SGD":         "s",
}

# ---------------------------------------------------------------------------
# Figure dimensions (inches) — NeurIPS single/double column
# ---------------------------------------------------------------------------
SINGLE_COL = (3.25, 2.5)
DOUBLE_COL = (6.75, 2.5)
DOUBLE_COL_TALL = (6.75, 3.5)


def setup_style():
    """Apply the paper-wide matplotlib rc settings."""
    mpl.rcParams.update({
        # Font
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
        "font.size":         8,
        "axes.labelsize":    9,
        "axes.titlesize":    9,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "legend.title_fontsize": 8,

        # Lines / markers
        "lines.linewidth":   1.2,
        "lines.markersize":  4,

        # Axes
        "axes.linewidth":    0.6,
        "axes.grid":         True,
        "axes.spines.top":   False,
        "axes.spines.right": False,

        # Grid
        "grid.color":        "#cccccc",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.4,
        "grid.alpha":        0.7,

        # Ticks
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.minor.size":  1.5,
        "ytick.minor.size":  1.5,

        # Figure
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,

        # Layout
        "figure.constrained_layout.use": True,

        # Math text
        "mathtext.fontset":  "cm",

        # Legend
        "legend.frameon":    True,
        "legend.framealpha": 0.85,
        "legend.edgecolor":  "#cccccc",
        "legend.fancybox":   False,
    })


def get_color(optimizer_name: str) -> str:
    """Return the canonical colour hex string for *optimizer_name*."""
    return COLORS.get(optimizer_name, "#333333")


def get_marker(optimizer_name: str) -> str:
    """Return the canonical marker character for *optimizer_name*."""
    return MARKERS.get(optimizer_name, "o")


def save_fig(fig, name: str, results_dir: str, formats=("pdf", "png")):
    """Save *fig* as ``<results_dir>/plots/<name>.{pdf,png}``."""
    out = Path(results_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out / f"{name}.{fmt}"
        fig.savefig(str(path), format=fmt)
        print(f"  Saved {path}")
    plt.close(fig)


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add a bold panel label like (a), (b), ... to an axes."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right")
