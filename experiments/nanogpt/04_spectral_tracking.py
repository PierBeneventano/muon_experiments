#!/usr/bin/env python3
"""
04_spectral_tracking.py -- Detailed spectral analysis.
Log SVD of EVERY weight matrix at --log_every intervals.
Save full singular-value vectors.

Usage:
  python experiments/nanogpt/04_spectral_tracking.py \
      --optimizer muon --seed 42 --log_every 100

SLURM: single GPU.
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


def parse_args():
    p = argparse.ArgumentParser(description="Detailed spectral tracking")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def default_lr(optimizer):
    return 0.02 if optimizer == "muon" else 1e-3


def run(args):
    lr = args.lr if args.lr is not None else default_lr(args.optimizer)
    tag = f"{args.optimizer}_every{args.log_every}_s{args.seed}"
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "results", "04_spectral_tracking", tag)
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

    meta = {
        "experiment": "04_spectral_tracking",
        "optimizer": args.optimizer,
        "lr": lr,
        "seed": args.seed,
        "log_every": args.log_every,
        "full_svd": True,
        "max_iters": args.max_iters,
    }

    print(f"[04] Spectral tracking: {args.optimizer} | log_every={args.log_every} | seed={args.seed}")
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

    # Count spectral snapshots
    spectral_log = os.path.join(out_dir, "spectral", "spectral_log.jsonl")
    n_snapshots = 0
    if os.path.isfile(spectral_log):
        with open(spectral_log) as f:
            n_snapshots = sum(1 for _ in f)
    meta["n_spectral_snapshots"] = n_snapshots

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[04] Done in {elapsed:.1f}s. {n_snapshots} snapshots -> {summary_path}")
    return meta


if __name__ == "__main__":
    run(parse_args())
