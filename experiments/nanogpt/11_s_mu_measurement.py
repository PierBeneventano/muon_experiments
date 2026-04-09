#!/usr/bin/env python3
"""
11_s_mu_measurement.py -- Load trained checkpoints and compute S(mu) for each
weight matrix. Compare across optimizers.

This is a post-hoc analysis script: it loads checkpoints saved by earlier
experiments and computes the S(mu) metric at multiple mu thresholds.

Usage:
  python experiments/nanogpt/11_s_mu_measurement.py \
      --checkpoint_dir results/01_muon_vs_adamw/muon_lr0.02_s42 \
      --output_dir results/11_s_mu_measurement/muon_s42

  # Or batch-compare two checkpoints:
  python experiments/nanogpt/11_s_mu_measurement.py \
      --checkpoint_dir results/01_muon_vs_adamw/muon_lr0.02_s42 \
      --compare_dir results/01_muon_vs_adamw/adamw_lr0.001_s42 \
      --output_dir results/11_s_mu_measurement/comparison_s42
"""

import argparse
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# We need torch; add nanoGPT to path for model loading
sys.path.insert(0, os.path.join(PROJECT_ROOT, "nanoGPT"))


def parse_args():
    p = argparse.ArgumentParser(description="S(mu) measurement from checkpoints")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Directory containing ckpt.pt from a training run")
    p.add_argument("--compare_dir", type=str, default=None,
                   help="Optional second checkpoint dir for comparison")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--mu_values", type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 2.0, 5.0],
                   help="Threshold values for S(mu) computation")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def load_checkpoint(ckpt_dir, device="cpu"):
    """Load a nanoGPT checkpoint and return model state dict."""
    import torch
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    return checkpoint


def compute_s_mu(singular_values, mu):
    """S(mu) = sum_i max(0, s_i - mu) / sum_i s_i."""
    s = singular_values.clamp(min=0)
    total = s.sum().item()
    if total < 1e-12:
        return 0.0
    return (s - mu).clamp(min=0).sum().item() / total


def spectral_entropy(singular_values):
    """Normalised spectral entropy."""
    s = singular_values.clamp(min=1e-12)
    p = s / s.sum()
    h = -(p * p.log()).sum().item()
    h_max = math.log(len(s)) if len(s) > 1 else 1.0
    return h / h_max if h_max > 0 else 0.0


def analyze_checkpoint(ckpt_dir, mu_values, device="cpu"):
    """Compute S(mu) for all weight matrices in a checkpoint."""
    import torch
    checkpoint = load_checkpoint(ckpt_dir, device)

    # state_dict may be under 'model' key
    state_dict = checkpoint.get("model", checkpoint)

    results = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() < 2:
            continue
        S = torch.linalg.svdvals(tensor.float().to(device))
        rec = {
            "name": name,
            "shape": list(tensor.shape),
            "spectral_entropy": round(spectral_entropy(S), 6),
            "top1_sv": round(S[0].item(), 6),
            "effective_rank": int((S > 1e-6).sum().item()),
            "frobenius_norm": round(tensor.float().norm().item(), 6),
        }
        for mu in mu_values:
            rec[f"s_mu_{mu}"] = round(compute_s_mu(S, mu), 6)
        rec["singular_values_top10"] = [round(v, 6) for v in S[:10].tolist()]
        results.append(rec)

    meta_info = {}
    if "config" in checkpoint:
        meta_info["config"] = checkpoint["config"]
    if "iter_num" in checkpoint:
        meta_info["iter_num"] = checkpoint["iter_num"]
    if "best_val_loss" in checkpoint:
        meta_info["best_val_loss"] = checkpoint["best_val_loss"]

    return results, meta_info


def run(args):
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "results", "11_s_mu_measurement", "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[11] Analyzing checkpoint: {args.checkpoint_dir}")
    results_a, meta_a = analyze_checkpoint(
        args.checkpoint_dir, args.mu_values, args.device)

    output = {
        "experiment": "11_s_mu_measurement",
        "mu_values": args.mu_values,
        "checkpoint_a": {
            "dir": args.checkpoint_dir,
            "meta": meta_a,
            "layers": results_a,
        },
    }

    # Print summary
    print(f"\n[11] Checkpoint A ({args.checkpoint_dir}):")
    print(f"     {'Layer':<40} {'H(s)':<8} " +
          " ".join(f"S({mu})" for mu in args.mu_values))
    for rec in results_a:
        row = f"     {rec['name']:<40} {rec['spectral_entropy']:<8.4f} "
        row += " ".join(f"{rec[f's_mu_{mu}']:<6.4f}" for mu in args.mu_values)
        print(row)

    if args.compare_dir:
        print(f"\n[11] Analyzing comparison checkpoint: {args.compare_dir}")
        results_b, meta_b = analyze_checkpoint(
            args.compare_dir, args.mu_values, args.device)
        output["checkpoint_b"] = {
            "dir": args.compare_dir,
            "meta": meta_b,
            "layers": results_b,
        }

        # Compute deltas
        name_to_b = {r["name"]: r for r in results_b}
        deltas = []
        for rec_a in results_a:
            rec_b = name_to_b.get(rec_a["name"])
            if rec_b is None:
                continue
            delta = {
                "name": rec_a["name"],
                "delta_entropy": round(rec_a["spectral_entropy"] - rec_b["spectral_entropy"], 6),
            }
            for mu in args.mu_values:
                key = f"s_mu_{mu}"
                delta[f"delta_{key}"] = round(rec_a[key] - rec_b[key], 6)
            deltas.append(delta)
        output["deltas_a_minus_b"] = deltas

        print(f"\n[11] Deltas (A - B):")
        for d in deltas:
            print(f"     {d['name']:<40} dH={d['delta_entropy']:+.4f}")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[11] Summary -> {summary_path}")
    return output


if __name__ == "__main__":
    run(parse_args())
