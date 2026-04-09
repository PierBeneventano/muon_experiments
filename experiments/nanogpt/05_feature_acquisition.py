#!/usr/bin/env python3
"""
05_feature_acquisition.py -- Track top-k singular values over training.
Monitor how quickly each singular value reaches its final magnitude.
Compare concurrent (Muon) vs sequential (AdamW) acquisition patterns.

Reads the full-SVD spectral log produced by 04_spectral_tracking.py or
by a training run with --spectral_full_svd=True, then computes per-SV
acquisition curves and summary statistics.

Usage:
  # Step 1: Run training with full SVD logging
  python experiments/nanogpt/05_feature_acquisition.py \
      --optimizer muon --seed 42 --log_every 100

  # Step 2 (analysis-only, if training was already done):
  python experiments/nanogpt/05_feature_acquisition.py \
      --analyze_only --spectral_log results/04_spectral_tracking/muon_every100_s42/spectral/spectral_log.jsonl \
      --output_dir results/05_feature_acquisition/muon_s42
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TRAIN_PY = os.path.join(PROJECT_ROOT, "nanoGPT", "train.py")


def parse_args():
    p = argparse.ArgumentParser(description="Feature acquisition analysis")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of top singular values to track")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    # Analysis-only mode
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip training, just analyze an existing spectral log")
    p.add_argument("--spectral_log", type=str, default=None,
                   help="Path to spectral_log.jsonl for analysis-only mode")
    return p.parse_args()


def default_lr(optimizer):
    return 0.02 if optimizer == "muon" else 1e-3


def analyze_acquisition(spectral_log_path, top_k=10):
    """
    Analyze feature-acquisition patterns from a full-SVD spectral log.
    Returns per-layer acquisition curves and summary statistics.
    """
    snapshots = []
    with open(spectral_log_path) as f:
        for line in f:
            snapshots.append(json.loads(line))

    if not snapshots:
        return {"error": "empty spectral log"}

    # Group by layer name
    layer_names = [rec["name"] for rec in snapshots[-1]["layers"]
                   if "singular_values" in rec]
    if not layer_names:
        return {"error": "no singular_values in log (run with spectral_full_svd=True)"}

    results = {}
    for lname in layer_names:
        sv_over_time = []  # list of (iter, [s0, s1, ...])
        for snap in snapshots:
            for rec in snap["layers"]:
                if rec["name"] == lname and "singular_values" in rec:
                    sv_over_time.append((snap["iter"], rec["singular_values"]))
                    break

        if not sv_over_time:
            continue

        # Final singular values
        final_svs = sv_over_time[-1][1]
        k = min(top_k, len(final_svs))

        # For each of the top-k SVs, compute:
        #   - acquisition curve: fraction of final value at each snapshot
        #   - t_90: first iteration where SV reaches 90% of its final value
        acquisition = []
        for i in range(k):
            final_val = final_svs[i]
            curve = []
            t_90 = None
            for it, svs in sv_over_time:
                val = svs[i] if i < len(svs) else 0.0
                frac = val / final_val if abs(final_val) > 1e-12 else 0.0
                curve.append({"iter": it, "sv": val, "fraction_of_final": round(frac, 4)})
                if t_90 is None and frac >= 0.9:
                    t_90 = it
            acquisition.append({
                "sv_index": i,
                "final_value": round(final_val, 6),
                "t_90": t_90,
                "curve": curve,
            })

        # Concurrency metric: std of t_90 values.
        # Low std = concurrent acquisition (Muon), high std = sequential (AdamW)
        t90_values = [a["t_90"] for a in acquisition if a["t_90"] is not None]
        if len(t90_values) >= 2:
            mean_t90 = sum(t90_values) / len(t90_values)
            std_t90 = math.sqrt(sum((t - mean_t90) ** 2 for t in t90_values) / len(t90_values))
        else:
            mean_t90 = t90_values[0] if t90_values else None
            std_t90 = 0.0

        results[lname] = {
            "top_k": k,
            "acquisition": acquisition,
            "mean_t90": round(mean_t90, 1) if mean_t90 is not None else None,
            "std_t90": round(std_t90, 1),
            "concurrency_score": round(1.0 / (1.0 + std_t90), 4) if std_t90 is not None else None,
        }

    return results


def run(args):
    lr = args.lr if args.lr is not None else default_lr(args.optimizer)
    tag = f"{args.optimizer}_topk{args.top_k}_s{args.seed}"
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "results", "05_feature_acquisition", tag)
    os.makedirs(out_dir, exist_ok=True)

    spectral_log_path = args.spectral_log

    if not args.analyze_only:
        # Run training with full SVD
        cmd = [
            sys.executable, TRAIN_PY,
            f"--out_dir={out_dir}",
            f"--seed={args.seed}",
            f"--device={args.device}",
            f"--dataset=shakespeare_char",
            f"--n_layer=4", f"--n_head=4", f"--n_embd=128",
            f"--block_size=256", f"--batch_size=64",
            f"--max_iters={args.max_iters}",
            f"--eval_interval=500",
            f"--log_interval=10",
            f"--eval_iters=200",
            f"--learning_rate={lr}",
            f"--muon_optimizer={args.optimizer}",
            f"--muon_lr={lr}",
            f"--spectral_log_every={args.log_every}",
            f"--spectral_full_svd=True",
            f"--wandb_log=False",
            f"--compile=False",
        f"--decay_lr=False",
        f"--gradient_accumulation_steps=1",
        f"--warmup_iters=100",
        ]

        print(f"[05] Training: {args.optimizer} | seed={args.seed}")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=os.path.join(PROJECT_ROOT, "nanoGPT"))
        elapsed = time.time() - t0
        print(f"[05] Training done in {elapsed:.1f}s (rc={result.returncode})")

        with open(os.path.join(out_dir, "stdout.txt"), "w") as f:
            f.write(result.stdout)
        with open(os.path.join(out_dir, "stderr.txt"), "w") as f:
            f.write(result.stderr)

        spectral_log_path = os.path.join(out_dir, "spectral", "spectral_log.jsonl")

    if spectral_log_path is None or not os.path.isfile(spectral_log_path):
        print("[05] ERROR: No spectral log found. Run training first or specify --spectral_log.")
        return None

    print(f"[05] Analyzing acquisition from {spectral_log_path}")
    analysis = analyze_acquisition(spectral_log_path, top_k=args.top_k)

    meta = {
        "experiment": "05_feature_acquisition",
        "optimizer": args.optimizer,
        "seed": args.seed,
        "top_k": args.top_k,
        "spectral_log": spectral_log_path,
        "acquisition_analysis": analysis,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Print concurrency scores
    print(f"\n[05] Concurrency scores (1=concurrent, 0=sequential):")
    for lname, data in analysis.items():
        if isinstance(data, dict) and "concurrency_score" in data:
            print(f"  {lname}: {data['concurrency_score']}")

    print(f"[05] Summary -> {summary_path}")
    return meta


if __name__ == "__main__":
    run(parse_args())
