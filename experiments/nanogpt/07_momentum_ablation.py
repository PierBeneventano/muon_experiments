#!/usr/bin/env python3
"""
07_momentum_ablation.py -- Muon momentum ablation.
Sweep: --momentum {0, 0.5, 0.9, 0.95, 0.99} (Muon only).
How does momentum affect spectral properties?

Usage:
  python experiments/nanogpt/07_momentum_ablation.py \
      --momentum 0.95 --seed 42

SLURM: one (momentum, seed) pair per job.
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TRAIN_PY = os.path.join(PROJECT_ROOT, "nanoGPT", "train.py")

MOMENTA = [0.0, 0.5, 0.9, 0.95, 0.99]


def parse_args():
    p = argparse.ArgumentParser(description="Muon momentum ablation")
    p.add_argument("--momentum", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def run(args):
    tag = f"muon_mom{args.momentum}_s{args.seed}"
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "results", "07_momentum_ablation", tag)
    os.makedirs(out_dir, exist_ok=True)

    n_layer, n_head, n_embd, block_size = 4, 4, 128, 256

    cmd = [
        sys.executable, TRAIN_PY,
        f"--out_dir={out_dir}",
        f"--seed={args.seed}",
        f"--device={args.device}",
        f"--dataset=shakespeare_char",
        f"--n_layer={n_layer}",
        f"--n_head={n_head}",
        f"--n_embd={n_embd}",
        f"--block_size={block_size}",
        f"--batch_size=64",
        f"--max_iters={args.max_iters}",
        f"--eval_interval=500",
        f"--log_interval=10",
        f"--eval_iters=200",
        f"--learning_rate={args.lr}",
        f"--optimizer=muon",
        f"--muon_lr={args.lr}",
        f"--muon_momentum={args.momentum}",
        f"--spectral_log_every=500",
        f"--spectral_full_svd=False",
        f"--wandb_log=False",
        f"--compile=False",
        f"--decay_lr=False",
        f"--gradient_accumulation_steps=1",
        f"--warmup_iters=100",
    ]

    meta = {
        "experiment": "07_momentum_ablation",
        "optimizer": "muon",
        "momentum": args.momentum,
        "lr": args.lr,
        "seed": args.seed,
        "max_iters": args.max_iters,
    }

    print(f"[07] momentum={args.momentum} | seed={args.seed}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.join(PROJECT_ROOT, "nanoGPT"))
    elapsed = time.time() - t0
    meta["elapsed_s"] = round(elapsed, 1)
    meta["returncode"] = result.returncode

    with open(os.path.join(out_dir, "stdout.txt"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(out_dir, "stderr.txt"), "w") as f:
        f.write(result.stderr)

    train_losses = []
    for line in result.stdout.splitlines():
        if line.startswith("iter ") and "loss" in line:
            parts = line.split()
            try:
                it = int(parts[1].rstrip(":"))
                lv = float(parts[3].rstrip(","))
                train_losses.append({"iter": it, "loss": lv})
            except (IndexError, ValueError):
                pass
    meta["train_losses"] = train_losses
    if train_losses:
        meta["final_train_loss"] = train_losses[-1]["loss"]

    # Read spectral data
    spectral_log = os.path.join(out_dir, "spectral", "spectral_log.jsonl")
    spectral_summary = []
    if os.path.isfile(spectral_log):
        with open(spectral_log) as f:
            for line in f:
                snap = json.loads(line)
                avg_entropy = sum(r["spectral_entropy"] for r in snap["layers"]) / max(len(snap["layers"]), 1)
                avg_s_mu = sum(r["s_mu_1"] for r in snap["layers"]) / max(len(snap["layers"]), 1)
                spectral_summary.append({
                    "iter": snap["iter"],
                    "avg_spectral_entropy": round(avg_entropy, 4),
                    "avg_s_mu_1": round(avg_s_mu, 4),
                })
    meta["spectral_summary"] = spectral_summary

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[07] Done in {elapsed:.1f}s -> {summary_path}")
    return meta


if __name__ == "__main__":
    run(parse_args())
