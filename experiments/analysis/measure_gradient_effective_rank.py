#!/usr/bin/env python3
"""
measure_gradient_effective_rank.py
==================================

H6 measurement script: gradient effective rank for NanoGPT and (optionally) ViT
checkpoints.

Background
----------
Hypothesis H6 of the Cycle-6 Muon paper asserts that Muon's advantage is mediated
by *gradient effective rank*: language-pretraining gradients are dense and
high-rank, while small-vision gradients are not. This script measures the
gradient effective rank r_eff(G) = ||G||_F^2 / ||G||_2^2 (the stable rank) on
flattened per-parameter-matrix gradients, averaged across attention projections.

Data sources
------------
Primary: NanoGPT checkpoints produced by `experiments/nanogpt/01_muon_vs_adamw.py`.
Each run directory `experiments/results/nanogpt/01_muon_vs_adamw/{opt}_s{seed}/`
should contain:
    - `ckpt.pt`       (final weights; optimizer state; model config)
    - `summary.json`  (run metadata)
    - `spectral/spectral_log.jsonl`  (per-layer weight spectra every 500 iter)

What we compute
---------------
For each checkpoint, we load the model, reconstruct the training data iterator,
draw a single held-out batch, run a forward+backward pass, and for each
attention projection (q, k, v, proj) compute the stable rank of the gradient
tensor reshaped to its natural 2-D matrix form.

If `ckpt.pt` does NOT include the information needed to reconstruct the model
(e.g. architecture hyperparameters missing), the script falls back to reading
the weight-spectrum stable-rank column from `spectral/spectral_log.jsonl` — this
is a *weight* effective rank, not gradient, and the script prints a clear
caveat so the user does not confuse the two.

CLI
---
    python experiments/analysis/measure_gradient_effective_rank.py --help

Typical invocations:
    # Measure from NanoGPT exp01 checkpoints in the default location
    python experiments/analysis/measure_gradient_effective_rank.py

    # Point at a specific run tree
    python experiments/analysis/measure_gradient_effective_rank.py \\
        --runs-dir /path/to/experiments/results/nanogpt/01_muon_vs_adamw

    # Include ViT checkpoints (from H5 vision run)
    python experiments/analysis/measure_gradient_effective_rank.py \\
        --runs-dir experiments/results/nanogpt/01_muon_vs_adamw \\
        --vit-runs-dir experiments/results/vision/01_vit_cifar10

    # Weight-spectrum-only fallback (no torch required)
    python experiments/analysis/measure_gradient_effective_rank.py --weights-only

Output
------
Writes `experiments/results/analysis/h6_gradient_effective_rank.json` with
per-run / per-layer stable-rank statistics plus an aggregate `r_lang / r_vision`
ratio and a Wilcoxon rank-sum p-value (if both modalities are present).

Phase 5b entry
--------------
This script is pre-registered as a Phase 5b (experiment_track) artifact. It is
runnable standalone but the gradient-mode measurement requires the NanoGPT
training data and model module to be importable; if not, the script reports
graceful fallback status so Phase 5b can schedule the cluster re-extraction.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Constants / defaults
# ----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # experiments/analysis -> experiments -> project
DEFAULT_NANOGPT_RUNS = (
    PROJECT_ROOT / "experiments" / "results" / "nanogpt" / "01_muon_vs_adamw"
)
DEFAULT_VIT_RUNS = (
    PROJECT_ROOT / "experiments" / "results" / "vision" / "01_vit_cifar10"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "analysis"
    / "h6_gradient_effective_rank.json"
)

ATTN_KEYS = ("c_attn", "attn.c_attn", "q_proj", "k_proj", "v_proj", "out_proj")


# ----------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class LayerMeasurement:
    name: str
    shape: Tuple[int, ...]
    frobenius_norm_sq: float
    operator_norm_sq: float
    stable_rank: float  # r_eff = frob^2 / op^2
    mode: str  # "gradient" or "weight" (fallback)


@dataclasses.dataclass
class RunMeasurement:
    run_id: str
    optimizer: str
    seed: Optional[int]
    modality: str  # "language" or "vision"
    mode: str  # "gradient" or "weight"
    mean_stable_rank: float
    median_stable_rank: float
    per_layer: List[LayerMeasurement]
    notes: str = ""


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


def stable_rank_from_matrix(mat) -> Tuple[float, float, float]:
    """Return (frob^2, op^2, stable_rank) for a 2-D tensor-like object.

    Accepts numpy arrays or torch tensors. Uses a single full SVD only to
    obtain the operator norm, since r_eff = ||G||_F^2 / ||G||_2^2 needs only
    the top singular value.
    """
    try:
        import torch  # type: ignore

        if hasattr(mat, "detach") and hasattr(mat, "cpu"):
            t = mat.detach().cpu().float()
            fro_sq = float(torch.sum(t * t).item())
            # operator norm via top-1 SVD is cheaper than full SVD
            op = float(torch.linalg.matrix_norm(t, ord=2).item())
            op_sq = op * op
            if op_sq <= 0.0:
                return fro_sq, op_sq, float("nan")
            return fro_sq, op_sq, fro_sq / op_sq
    except Exception:
        pass

    import numpy as np

    arr = np.asarray(mat, dtype=np.float64)
    fro_sq = float(np.sum(arr * arr))
    if arr.ndim != 2:
        raise ValueError(f"stable_rank expects 2-D matrix, got shape {arr.shape}")
    # Top-1 SV via np.linalg.norm(..., 2) is O(mn min(m,n)); acceptable for
    # small attention projections (d<=768 typically).
    op = float(np.linalg.norm(arr, ord=2))
    op_sq = op * op
    if op_sq <= 0.0:
        return fro_sq, op_sq, float("nan")
    return fro_sq, op_sq, fro_sq / op_sq


def _reshape_grad_to_matrix(name: str, grad) -> Optional["object"]:
    """Flatten a gradient tensor to 2-D for stable-rank computation.

    Rules:
      - 2-D: return as-is.
      - 3-D+: reshape to (d0, prod(rest)).
      - 1-D: skip (biases, norms).
    """
    try:
        import torch  # type: ignore

        if isinstance(grad, torch.Tensor):
            if grad.ndim < 2:
                return None
            if grad.ndim == 2:
                return grad
            return grad.reshape(grad.shape[0], -1)
    except Exception:
        pass
    import numpy as np

    arr = np.asarray(grad)
    if arr.ndim < 2:
        return None
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _is_attention_param(name: str) -> bool:
    name_l = name.lower()
    return any(k in name_l for k in ATTN_KEYS) or (
        "attn" in name_l and "weight" in name_l and "ln" not in name_l
    )


# ----------------------------------------------------------------------------
# NanoGPT checkpoint discovery
# ----------------------------------------------------------------------------


def discover_nanogpt_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    out: List[Path] = []
    for child in sorted(runs_dir.iterdir()):
        if child.is_dir() and (child / "ckpt.pt").exists():
            out.append(child)
    return out


def _parse_run_tag(run_dir: Path) -> Tuple[str, Optional[int]]:
    """e.g. 'muon_s42' -> ('muon', 42)."""
    tag = run_dir.name
    parts = tag.split("_")
    opt = parts[0] if parts else "unknown"
    seed: Optional[int] = None
    for p in parts[1:]:
        if p.startswith("s") and p[1:].isdigit():
            seed = int(p[1:])
            break
    return opt, seed


# ----------------------------------------------------------------------------
# Gradient-mode measurement (requires torch + importable model)
# ----------------------------------------------------------------------------


def measure_gradient_mode(run_dir: Path, device: str = "cpu") -> Optional[RunMeasurement]:
    """Attempt full forward+backward on a saved NanoGPT checkpoint.

    Returns None with a printed reason if prerequisites are missing.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        print(f"  [gradient-mode] torch unavailable ({exc}); skipping {run_dir.name}")
        return None

    # Find the NanoGPT model module. We look for `nanoGPT/model.py` as the
    # canonical NanoGPT fork shipped with this project.
    nanogpt_candidates = [
        PROJECT_ROOT / "nanoGPT",
        PROJECT_ROOT.parent / "nanoGPT",
        Path.cwd() / "nanoGPT",
    ]
    nanogpt_dir: Optional[Path] = None
    for cand in nanogpt_candidates:
        if (cand / "model.py").exists():
            nanogpt_dir = cand
            break
    if nanogpt_dir is None:
        print(
            f"  [gradient-mode] nanoGPT/model.py not found; falling back to weight mode"
        )
        return None

    sys.path.insert(0, str(nanogpt_dir))
    try:
        from model import GPT, GPTConfig  # type: ignore
    except Exception as exc:
        print(f"  [gradient-mode] cannot import GPT from {nanogpt_dir}: {exc}")
        return None

    ckpt_path = run_dir / "ckpt.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"  [gradient-mode] torch.load failed on {ckpt_path}: {exc}")
        return None

    model_args = ckpt.get("model_args")
    if model_args is None:
        print(f"  [gradient-mode] checkpoint lacks model_args; skipping")
        return None

    cfg = GPTConfig(**model_args)
    model = GPT(cfg).to(device)
    state = ckpt.get("model")
    if state is None:
        print(f"  [gradient-mode] checkpoint lacks model state_dict; skipping")
        return None
    # Strip nanoGPT's '_orig_mod.' prefix that compile() adds
    clean = {}
    for k, v in state.items():
        clean[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(clean, strict=False)
    model.train()

    # Draw a held-out batch. nanoGPT Shakespeare-char data lives at
    # nanoGPT/data/shakespeare_char/{val.bin}. We read it as uint16 tokens.
    val_bin = nanogpt_dir / "data" / "shakespeare_char" / "val.bin"
    if not val_bin.exists():
        print(f"  [gradient-mode] val data missing at {val_bin}; skipping")
        return None

    import numpy as np

    data = np.memmap(str(val_bin), dtype=np.uint16, mode="r")
    block = cfg.block_size
    bs = 8  # small held-out batch
    ix = np.random.default_rng(0).integers(0, len(data) - block - 1, size=bs)
    x = torch.stack(
        [torch.from_numpy(data[i : i + block].astype(np.int64)) for i in ix]
    ).to(device)
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block].astype(np.int64)) for i in ix]
    ).to(device)

    model.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()

    per_layer: List[LayerMeasurement] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not _is_attention_param(name):
            continue
        mat = _reshape_grad_to_matrix(name, p.grad)
        if mat is None:
            continue
        fro_sq, op_sq, sr = stable_rank_from_matrix(mat)
        per_layer.append(
            LayerMeasurement(
                name=name,
                shape=tuple(p.grad.shape),
                frobenius_norm_sq=fro_sq,
                operator_norm_sq=op_sq,
                stable_rank=sr,
                mode="gradient",
            )
        )

    if not per_layer:
        print(f"  [gradient-mode] no attention-matrix gradients captured in {run_dir.name}")
        return None

    opt, seed = _parse_run_tag(run_dir)
    srs = [L.stable_rank for L in per_layer if not math.isnan(L.stable_rank)]
    return RunMeasurement(
        run_id=run_dir.name,
        optimizer=opt,
        seed=seed,
        modality="language",
        mode="gradient",
        mean_stable_rank=float(sum(srs) / len(srs)),
        median_stable_rank=float(sorted(srs)[len(srs) // 2]),
        per_layer=per_layer,
        notes="gradient stable rank via single held-out batch forward+backward",
    )


# ----------------------------------------------------------------------------
# Weight-mode fallback (uses the existing per-run spectral_log.jsonl)
# ----------------------------------------------------------------------------


def measure_weight_mode(run_dir: Path) -> Optional[RunMeasurement]:
    """Fallback: use weight stable-rank at the final logged step from spectral_log.jsonl.

    This is NOT a gradient measurement; it is the same stable-rank metric but on
    the final weight matrices. We use it only to keep the script runnable as a
    standalone sanity check when gradient mode is unavailable. The output is
    flagged `mode="weight"` so downstream analysis does not confuse the two.
    """
    log_path = run_dir / "spectral" / "spectral_log.jsonl"
    if not log_path.exists():
        return None

    last_line: Optional[str] = None
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    try:
        rec = json.loads(last_line)
    except Exception:
        return None

    per_layer: List[LayerMeasurement] = []
    for layer in rec.get("layers", []):
        name = layer.get("name", "")
        if not _is_attention_param(name):
            continue
        shape = tuple(layer.get("shape", ()))
        sr = layer.get("stable_rank")
        fro = layer.get("frobenius_norm")
        op = layer.get("operator_norm")
        if sr is None or fro is None or op is None:
            continue
        per_layer.append(
            LayerMeasurement(
                name=name,
                shape=shape,
                frobenius_norm_sq=float(fro) ** 2,
                operator_norm_sq=float(op) ** 2,
                stable_rank=float(sr),
                mode="weight",
            )
        )

    if not per_layer:
        return None

    opt, seed = _parse_run_tag(run_dir)
    srs = [L.stable_rank for L in per_layer if not math.isnan(L.stable_rank)]
    return RunMeasurement(
        run_id=run_dir.name,
        optimizer=opt,
        seed=seed,
        modality="language",
        mode="weight",
        mean_stable_rank=float(sum(srs) / len(srs)),
        median_stable_rank=float(sorted(srs)[len(srs) // 2]),
        per_layer=per_layer,
        notes=(
            "WEIGHT stable rank (fallback — not a gradient measurement). "
            "To compute gradient effective rank, rerun with NanoGPT importable."
        ),
    )


# ----------------------------------------------------------------------------
# ViT (H5) measurement, mirrors NanoGPT path
# ----------------------------------------------------------------------------


def measure_vit_gradient(run_dir: Path, device: str = "cpu") -> Optional[RunMeasurement]:
    """ViT checkpoint gradient stable-rank measurement (for H5 paired runs).

    Follows the same protocol as measure_gradient_mode: loads the ViT model
    from `train_vit_cifar.py`-produced checkpoint, draws a single CIFAR-10
    batch, backprops the cross-entropy loss, and computes per-attention-proj
    gradient stable ranks. Falls back to a skip with explanation if the
    checkpoint / dataset / model module is unavailable.
    """
    ckpt_path = run_dir / "ckpt.pt"
    if not ckpt_path.exists():
        return None

    try:
        import torch  # type: ignore
    except Exception:
        return None

    vit_script = PROJECT_ROOT / "experiments" / "vision" / "train_vit_cifar.py"
    if not vit_script.exists():
        print(f"  [vit] {vit_script} not found; skipping")
        return None

    sys.path.insert(0, str(vit_script.parent))
    try:
        import train_vit_cifar as vit_mod  # type: ignore
    except Exception as exc:
        print(f"  [vit] cannot import train_vit_cifar: {exc}")
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"  [vit] torch.load failed: {exc}")
        return None

    build_fn = getattr(vit_mod, "build_model", None)
    if build_fn is None:
        print(f"  [vit] train_vit_cifar.py exposes no build_model(); skipping")
        return None

    model = build_fn(**ckpt.get("model_args", {})).to(device)
    state = ckpt.get("model") or ckpt.get("state_dict")
    if state is None:
        print(f"  [vit] checkpoint lacks model state; skipping")
        return None
    model.load_state_dict(state, strict=False)
    model.train()

    # Held-out CIFAR-10 batch
    try:
        from torchvision import datasets, transforms  # type: ignore

        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        ds = datasets.CIFAR10(
            root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=tf
        )
        x_list, y_list = [], []
        for i in range(8):
            xi, yi = ds[i]
            x_list.append(xi)
            y_list.append(yi)
        x = torch.stack(x_list).to(device)
        y = torch.tensor(y_list).to(device)
    except Exception as exc:
        print(f"  [vit] dataset load failed: {exc}")
        return None

    model.zero_grad(set_to_none=True)
    out = model(x)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(out, y)
    loss.backward()

    per_layer: List[LayerMeasurement] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not _is_attention_param(name):
            continue
        mat = _reshape_grad_to_matrix(name, p.grad)
        if mat is None:
            continue
        fro_sq, op_sq, sr = stable_rank_from_matrix(mat)
        per_layer.append(
            LayerMeasurement(
                name=name,
                shape=tuple(p.grad.shape),
                frobenius_norm_sq=fro_sq,
                operator_norm_sq=op_sq,
                stable_rank=sr,
                mode="gradient",
            )
        )

    if not per_layer:
        return None

    opt, seed = _parse_run_tag(run_dir)
    srs = [L.stable_rank for L in per_layer if not math.isnan(L.stable_rank)]
    return RunMeasurement(
        run_id=run_dir.name,
        optimizer=opt,
        seed=seed,
        modality="vision",
        mode="gradient",
        mean_stable_rank=float(sum(srs) / len(srs)),
        median_stable_rank=float(sorted(srs)[len(srs) // 2]),
        per_layer=per_layer,
        notes="ViT CIFAR-10 gradient stable rank via single forward+backward",
    )


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------


def wilcoxon_rank_sum(a: List[float], b: List[float]) -> Optional[float]:
    """Rough Mann-Whitney U p-value approximation using scipy if available."""
    if not a or not b:
        return None
    try:
        from scipy.stats import mannwhitneyu  # type: ignore

        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return None


def aggregate(measurements: List[RunMeasurement]) -> Dict:
    lang = [m for m in measurements if m.modality == "language"]
    vis = [m for m in measurements if m.modality == "vision"]
    lang_srs = [m.mean_stable_rank for m in lang]
    vis_srs = [m.mean_stable_rank for m in vis]

    def _mean(xs: Iterable[float]) -> Optional[float]:
        xs = list(xs)
        return float(sum(xs) / len(xs)) if xs else None

    r_lang = _mean(lang_srs)
    r_vision = _mean(vis_srs)
    ratio = None
    if r_lang is not None and r_vision not in (None, 0.0):
        ratio = r_lang / r_vision

    return {
        "n_language_runs": len(lang),
        "n_vision_runs": len(vis),
        "mean_stable_rank_language": r_lang,
        "mean_stable_rank_vision": r_vision,
        "ratio_lang_over_vision": ratio,
        "wilcoxon_p_two_sided": wilcoxon_rank_sum(lang_srs, vis_srs),
        "mode_counts": {
            "gradient": sum(1 for m in measurements if m.mode == "gradient"),
            "weight": sum(1 for m in measurements if m.mode == "weight"),
        },
    }


def measurement_to_dict(m: RunMeasurement) -> Dict:
    return {
        "run_id": m.run_id,
        "optimizer": m.optimizer,
        "seed": m.seed,
        "modality": m.modality,
        "mode": m.mode,
        "mean_stable_rank": m.mean_stable_rank,
        "median_stable_rank": m.median_stable_rank,
        "notes": m.notes,
        "per_layer": [
            {
                "name": L.name,
                "shape": list(L.shape),
                "stable_rank": L.stable_rank,
                "mode": L.mode,
            }
            for L in m.per_layer
        ],
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Measure gradient effective rank (stable rank) for NanoGPT / ViT "
            "checkpoints. H6 of project_003_muon."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_NANOGPT_RUNS,
        help=f"NanoGPT runs directory (default: {DEFAULT_NANOGPT_RUNS})",
    )
    p.add_argument(
        "--vit-runs-dir",
        type=Path,
        default=None,
        help=(
            f"ViT runs directory (from H5; default skipped, pass e.g. {DEFAULT_VIT_RUNS})"
        ),
    )
    p.add_argument(
        "--weights-only",
        action="store_true",
        help=(
            "Skip gradient-mode measurement; use spectral_log.jsonl weight "
            "stable rank only. Produces a WEIGHT-based report, not a gradient "
            "report. Useful as a dependency-light smoke test."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for gradient-mode forward/backward (default cpu)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = p.parse_args()

    runs_dir: Path = args.runs_dir
    print(f"[H6] Scanning {runs_dir}")
    nanogpt_runs = discover_nanogpt_runs(runs_dir)
    if not nanogpt_runs:
        print(
            f"[H6] No NanoGPT runs found under {runs_dir}. "
            f"Expected subdirs like muon_s42/ containing ckpt.pt."
        )

    measurements: List[RunMeasurement] = []
    for run_dir in nanogpt_runs:
        print(f"[H6] run {run_dir.name}")
        m: Optional[RunMeasurement] = None
        if not args.weights_only:
            m = measure_gradient_mode(run_dir, device=args.device)
        if m is None:
            m = measure_weight_mode(run_dir)
        if m is None:
            print(f"  -> no measurement possible")
            continue
        print(
            f"  -> modality={m.modality} mode={m.mode} "
            f"mean_stable_rank={m.mean_stable_rank:.3f} "
            f"n_layers={len(m.per_layer)}"
        )
        measurements.append(m)

    if args.vit_runs_dir is not None and args.vit_runs_dir.exists():
        print(f"[H6] Scanning ViT runs {args.vit_runs_dir}")
        for run_dir in sorted(args.vit_runs_dir.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "ckpt.pt").exists():
                continue
            print(f"[H6] ViT run {run_dir.name}")
            m = measure_vit_gradient(run_dir, device=args.device)
            if m is None:
                print(f"  -> ViT measurement unavailable")
                continue
            print(
                f"  -> modality=vision mode=gradient "
                f"mean_stable_rank={m.mean_stable_rank:.3f} "
                f"n_layers={len(m.per_layer)}"
            )
            measurements.append(m)
    elif args.vit_runs_dir is not None:
        print(
            f"[H6] ViT runs dir {args.vit_runs_dir} does not exist yet — "
            f"H5 vision runs required. Skipping ViT contribution."
        )

    agg = aggregate(measurements)
    report = {
        "script": "experiments/analysis/measure_gradient_effective_rank.py",
        "hypothesis": "H6",
        "definition": "r_eff(G) = ||G||_F^2 / ||G||_2^2 (stable rank)",
        "n_measurements": len(measurements),
        "aggregate": agg,
        "runs": [measurement_to_dict(m) for m in measurements],
        "caveats": [
            "Gradient-mode requires NanoGPT model module importable; otherwise "
            "the script falls back to WEIGHT stable rank from spectral_log.jsonl.",
            "A single held-out batch is used for the forward+backward pass; "
            "add seed-averaging via repeated invocation for tighter CIs.",
            "H6 pre-registration: statistic of interest is the ratio "
            "r_lang / r_vision with Wilcoxon rank-sum p-value; mode='weight' "
            "rows should be excluded from the final ratio if any 'gradient' "
            "rows are present on the language side.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[H6] wrote {args.output}")
    if not measurements:
        print(
            "[H6] NO MEASUREMENTS PRODUCED. H6 requires cluster execution with "
            "the added gradient-logging hook; this script should be re-run after "
            "Phase 5b schedules the NanoGPT re-extraction job."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
