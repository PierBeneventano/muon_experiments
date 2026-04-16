#!/usr/bin/env python3
"""
reconstruct_stable_rank_population.py
=====================================

G4 deliverable: reconstruct per-run stable-rank trajectories from the existing
on-disk ``spectral_log.jsonl`` files, across ALL completed NanoGPT runs under
``experiments/results/nanogpt/``. Zero new compute — pure aggregation of
already-logged per-layer spectra.

For every run directory that contains ``spectral/spectral_log.jsonl`` and a
``summary.json`` (with the ``optimizer`` field), we:

  1. Stream the spectral log;
  2. For each iteration snapshot, average the per-layer ``stable_rank`` field
     across weight matrices of the transformer blocks (skipping embeddings
     and LayerNorm 1-D params, in line with the H6/G3 convention);
  3. Record (iter, mean_stable_rank) trajectories per run;
  4. Aggregate into optimizer-level population mean +/- 1 std bands over the
     full population of runs (N ~= 106 on disk at time of writing; target 150
     when exp02/03 spectral re-extractions land).

Outputs
-------
- ``experiments/results/analysis/150run_stable_rank_trajectories.json``
  with schema::
      {
        "source_dir": "...",
        "n_runs_found": int,
        "n_runs_kept":  int,
        "skipped":      [{"run": "...", "reason": "..."}, ...],
        "per_run": [
           {"run": "<exp>/<tag>", "optimizer": "muon"|"adamw",
            "seed": int|None, "experiment": "...",
            "iters": [...], "stable_rank_mean": [...]}
        ],
        "population": {
           "adamw": {"iters":[...], "mean":[...], "std":[...], "n_runs":int},
           "muon":  {"iters":[...], "mean":[...], "std":[...], "n_runs":int}
        }
      }

- ``experiments/results/plots/fig_150run_stable_rank.pdf`` + ``.png``

The figure follows ``plots/style.py`` conventions (Wong palette, NeurIPS
single-column dimensions, save_fig wrapper).

Usage
-----
    python experiments/analysis/reconstruct_stable_rank_population.py \\
        --results_dir experiments/results/nanogpt/ \\
        --output_json experiments/results/analysis/150run_stable_rank_trajectories.json \\
        --output_fig  experiments/results/plots/fig_150run_stable_rank.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib is required for the figure; import lazily so the JSON can still
# be produced even if a headless env lacks it.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Try to reuse the paper-wide style utilities; fall back gracefully.
_STYLE_OK = True
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from plots.style import (  # type: ignore
        setup_style,
        get_color,
        save_fig,
        SINGLE_COL,
    )
except Exception:
    _STYLE_OK = False


# ---------------------------------------------------------------------------
# Layer filtering
# ---------------------------------------------------------------------------
# Stable rank is only meaningful for 2-D weight matrices inside transformer
# blocks. We skip:
#   - token/position embeddings (tied / low-rank by construction),
#   - LayerNorm and bias (1-D),
#   - the final LM head if it is tied to wte (shape (vocab, d_model)).
_SKIP_SUBSTRINGS = ("wte", "wpe", "ln_", "bias", "lm_head")


def _keep_layer(layer: dict) -> bool:
    name = layer.get("name", "")
    shape = layer.get("shape", [])
    if len(shape) != 2:
        return False
    for s in _SKIP_SUBSTRINGS:
        if s in name:
            return False
    return True


def _parse_run(run_dir: Path) -> Optional[dict]:
    """Return a trajectory dict for *run_dir* or None if it should be skipped."""
    summary_path = run_dir / "summary.json"
    spectral_path = run_dir / "spectral" / "spectral_log.jsonl"
    if not summary_path.exists() or not spectral_path.exists():
        return None

    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except Exception:
        return None

    optimizer = summary.get("optimizer")
    if optimizer not in ("muon", "adamw"):
        return None
    seed = summary.get("seed")
    experiment = summary.get("experiment", run_dir.parent.name)

    iters: List[int] = []
    sr_means: List[float] = []
    with open(spectral_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            layers = record.get("layers", [])
            vals = [
                float(l["stable_rank"])
                for l in layers
                if _keep_layer(l) and "stable_rank" in l
            ]
            if not vals:
                continue
            iters.append(int(record.get("iter", len(iters))))
            sr_means.append(float(np.mean(vals)))

    if len(iters) < 2:
        return None

    return {
        "run": f"{run_dir.parent.name}/{run_dir.name}",
        "optimizer": optimizer,
        "seed": seed,
        "experiment": experiment,
        "iters": iters,
        "stable_rank_mean": sr_means,
    }


# ---------------------------------------------------------------------------
# Population aggregation
# ---------------------------------------------------------------------------

def _aggregate_population(per_run: List[dict]) -> Dict[str, dict]:
    """Interpolate every run onto a shared iteration grid, per optimizer."""
    out: Dict[str, dict] = {}
    by_opt: Dict[str, List[dict]] = defaultdict(list)
    for r in per_run:
        by_opt[r["optimizer"]].append(r)

    for opt, runs in by_opt.items():
        # Common grid: coarsest per-run sampling -> take the median max iter,
        # 30 grid points; interpolate each run onto it.
        max_iters = [r["iters"][-1] for r in runs]
        target_max = int(np.median(max_iters)) if max_iters else 0
        if target_max <= 0:
            continue
        grid = np.linspace(0, target_max, 30)

        stacked = []
        for r in runs:
            xs = np.asarray(r["iters"], dtype=float)
            ys = np.asarray(r["stable_rank_mean"], dtype=float)
            if xs[-1] < target_max * 0.5:
                # truncated run -- skip from population curve
                continue
            interp = np.interp(grid, xs, ys, left=np.nan, right=np.nan)
            stacked.append(interp)

        if not stacked:
            continue
        arr = np.vstack(stacked)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        out[opt] = {
            "iters": grid.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "n_runs": int(arr.shape[0]),
        }
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _plot(per_run: List[dict], population: Dict[str, dict], out_path: Path) -> None:
    if plt is None:
        print("[G4] matplotlib unavailable; skipping figure.")
        return
    if _STYLE_OK:
        setup_style()
        fig, ax = plt.subplots(figsize=SINGLE_COL)
    else:
        fig, ax = plt.subplots(figsize=(3.25, 2.5))

    def _color(opt: str) -> str:
        if _STYLE_OK:
            return get_color("Muon" if opt == "muon" else "AdamW")
        return "#0072B2" if opt == "muon" else "#CC79A7"

    # Individual-run thin lines (alpha low) for context.
    for r in per_run:
        ax.plot(
            r["iters"],
            r["stable_rank_mean"],
            color=_color(r["optimizer"]),
            alpha=0.08,
            linewidth=0.5,
            zorder=1,
        )

    # Population mean +/- 1 std bands.
    for opt, agg in population.items():
        g = np.asarray(agg["iters"])
        m = np.asarray(agg["mean"])
        s = np.asarray(agg["std"])
        label_opt = "Muon" if opt == "muon" else "AdamW"
        ax.plot(
            g,
            m,
            color=_color(opt),
            linewidth=1.6,
            label=f"{label_opt} (N={agg['n_runs']})",
            zorder=3,
        )
        ax.fill_between(
            g,
            m - s,
            m + s,
            color=_color(opt),
            alpha=0.25,
            linewidth=0,
            zorder=2,
        )

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean block stable rank")
    ax.set_title("Stable-rank trajectories: full NanoGPT run population")
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _STYLE_OK:
        # save_fig expects results_dir/plots/<name>
        name = out_path.stem
        results_dir = out_path.parent.parent
        save_fig(fig, name, str(results_dir))
    else:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[G4] Figure saved near {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("experiments/results/nanogpt/"),
        help="NanoGPT results root (default: experiments/results/nanogpt/)",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=Path("experiments/results/analysis/150run_stable_rank_trajectories.json"),
    )
    p.add_argument(
        "--output_fig",
        type=Path,
        default=Path("experiments/results/plots/fig_150run_stable_rank.pdf"),
    )
    p.add_argument(
        "--min_runs_for_success",
        type=int,
        default=120,
        help="Minimum number of runs parsed to satisfy G4 minimum-viable (default 120).",
    )
    args = p.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"[G4] results_dir does not exist: {results_dir}", file=sys.stderr)
        return 2

    per_run: List[dict] = []
    skipped: List[dict] = []
    n_found = 0
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            n_found += 1
            trajectory = _parse_run(run_dir)
            if trajectory is None:
                skipped.append({
                    "run": f"{exp_dir.name}/{run_dir.name}",
                    "reason": "missing_log_or_unrecognised_optimizer",
                })
            else:
                per_run.append(trajectory)

    population = _aggregate_population(per_run)

    payload = {
        "source_dir": str(results_dir),
        "n_runs_found": n_found,
        "n_runs_kept": len(per_run),
        "skipped": skipped,
        "per_run": per_run,
        "population": population,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[G4] Kept {len(per_run)}/{n_found} runs; wrote {args.output_json}"
    )

    _plot(per_run, population, args.output_fig)

    # Self-assessment w.r.t. G4 success gate.
    ok = len(per_run) >= args.min_runs_for_success
    # Separation check: AdamW end < Muon end.
    end_sep_ok = False
    if "adamw" in population and "muon" in population:
        a_end = population["adamw"]["mean"][-1]
        m_end = population["muon"]["mean"][-1]
        end_sep_ok = a_end < m_end
    print(
        f"[G4] coverage_pass={ok} ({len(per_run)} / {args.min_runs_for_success}) "
        f"end_separation_pass={end_sep_ok}"
    )
    return 0 if (ok and end_sep_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
