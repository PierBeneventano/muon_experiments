#!/usr/bin/env python3
"""
Master plot script — runs every individual plot script and reports status.

Usage:
    python experiments/plots/plot_all.py \
        --results_dir experiments/results \
        --output_dir  experiments/results
"""

import argparse
import importlib
import sys
import traceback
from pathlib import Path

# All plot modules in figure order
PLOT_MODULES = [
    ("plots.plot_e1_hero",           "Figure 1  — Hero (spectra + entropy)"),
    ("plots.plot_kappa_scaling",     "Figure 3  — Kappa scaling"),
    ("plots.plot_alignment",         "Figure 4  — Alignment dynamics"),
    ("plots.plot_block_acquisition", "Figure 5  — Block acquisition"),
    ("plots.plot_nanogpt_bcrit",     "Figure 6  — NanoGPT B_crit"),
    ("plots.plot_nanogpt_spectral",  "Figure 7  — NanoGPT spectral"),
    ("plots.plot_1500_ablation",     "Figure 8  — 1500-config ablation"),
    ("plots.plot_taxonomy",          "Table  1  — Taxonomy"),
]


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Root results directory (contains matrix_sensing/, nanogpt/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory (plots saved to <output_dir>/plots/)")
    args = parser.parse_args()

    # Ensure the plots package is importable
    exp_root = str(Path(__file__).resolve().parent.parent)
    if exp_root not in sys.path:
        sys.path.insert(0, exp_root)

    passed, failed, skipped = 0, 0, 0

    print("=" * 60)
    print("  Generating all figures")
    print(f"  Results : {args.results_dir}")
    print(f"  Output  : {args.output_dir}")
    print("=" * 60)

    for mod_name, description in PLOT_MODULES:
        print(f"\n[PLOT] {description} ... ", end="", flush=True)
        try:
            mod = importlib.import_module(mod_name)
            mod.plot(args.results_dir, args.output_dir)
            print("OK")
            passed += 1
        except FileNotFoundError as exc:
            print(f"SKIPPED (missing data: {exc.filename})")
            skipped += 1
        except Exception:
            print("FAILED")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {skipped} skipped, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
