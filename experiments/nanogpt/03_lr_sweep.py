#!/usr/bin/env python3
"""
03_lr_sweep.py -- LR plateau detection.
Grid: 7 LRs x 7 batch sizes x 2 optimizers = 98 configs, 3000 iterations each.

Usage (single point in the grid):
  python experiments/nanogpt/03_lr_sweep.py \
      --lr 0.003 --batch_size 64 --optimizer muon --seed 42

SLURM: one (lr, batch_size, optimizer, seed) combo per job.
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

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]


def parse_args():
    p = argparse.ArgumentParser(description="LR sweep (plateau detection)")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="adamw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def run(args):
    tag = f"{args.optimizer}_lr{args.lr}_bs{args.batch_size}_s{args.seed}"
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "results", "03_lr_sweep", tag)
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
        f"--batch_size={args.batch_size}",
        f"--max_iters={args.max_iters}",
        f"--eval_interval=500",
        f"--log_interval=10",
        f"--eval_iters=200",
        f"--learning_rate={args.lr}",
        f"--optimizer={args.optimizer}",
        f"--muon_lr={args.lr}",
        f"--spectral_log_every=0",  # no spectral logging -- too many runs
        f"--wandb_log=False",
        f"--compile=False",
    ]

    meta = {
        "experiment": "03_lr_sweep",
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_iters": args.max_iters,
    }

    print(f"[03] lr={args.lr} | bs={args.batch_size} | {args.optimizer} | seed={args.seed}")
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

    # Extract final train loss for quick comparison
    if train_losses:
        meta["final_train_loss"] = train_losses[-1]["loss"]

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[03] Done in {elapsed:.1f}s -> {summary_path}")
    return meta


if __name__ == "__main__":
    run(parse_args())
