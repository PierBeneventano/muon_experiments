#!/usr/bin/env python3
"""
02_batch_size_sweep.py -- B_crit validation: sweep batch sizes for both optimizers.
5000 iterations, same model config as 01.

Usage:
  python experiments/nanogpt/02_batch_size_sweep.py \
      --batch_size 64 --optimizer muon --seed 42

SLURM: one (batch_size, optimizer, seed) triple per job.
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

BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]


def parse_args():
    p = argparse.ArgumentParser(description="Batch-size sweep for B_crit validation")
    p.add_argument("--batch_size", type=int, default=64, choices=BATCH_SIZES)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="adamw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def default_lr(optimizer):
    return 0.02 if optimizer == "muon" else 1e-3


def run(args):
    lr = args.lr if args.lr is not None else default_lr(args.optimizer)
    tag = f"{args.optimizer}_bs{args.batch_size}_s{args.seed}"
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "results", "02_batch_size_sweep", tag)
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
        f"--learning_rate={lr}",
        f"--optimizer={args.optimizer}",
        f"--muon_lr={lr}",
        f"--spectral_log_every=500",
        f"--wandb_log=False",
        f"--compile=False",
        f"--decay_lr=False",
        f"--gradient_accumulation_steps=1",
        f"--warmup_iters=100",
    ]

    meta = {
        "experiment": "02_batch_size_sweep",
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "lr": lr,
        "seed": args.seed,
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "block_size": block_size,
        "max_iters": args.max_iters,
    }

    print(f"[02] batch_size={args.batch_size} | {args.optimizer} | seed={args.seed}")
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

    # Parse losses
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

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[02] Done in {elapsed:.1f}s -> {summary_path}")
    return meta


if __name__ == "__main__":
    run(parse_args())
