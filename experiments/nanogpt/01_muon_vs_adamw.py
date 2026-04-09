#!/usr/bin/env python3
"""
01_muon_vs_adamw.py -- Main comparison: Muon vs AdamW on Shakespeare char-level.
5000 iterations, spectral entropy logged every 500 steps.

Usage:
  python experiments/nanogpt/01_muon_vs_adamw.py \
      --optimizer muon --seed 42 --lr 0.02 --output_dir results/01/muon_s42

SLURM-compatible: single GPU per job.
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
    p = argparse.ArgumentParser(description="Muon vs AdamW comparison")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="adamw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 1e-3 for adamw, 0.02 for muon)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config to load base settings from")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def default_lr(optimizer: str) -> float:
    return 0.02 if optimizer == "muon" else 1e-3


def build_output_dir(args) -> str:
    if args.output_dir:
        return args.output_dir
    tag = f"{args.optimizer}_lr{args.lr}_s{args.seed}"
    return os.path.join(PROJECT_ROOT, "results", "01_muon_vs_adamw", tag)


def run(args):
    lr = args.lr if args.lr is not None else default_lr(args.optimizer)
    args.lr = lr
    out_dir = build_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    # Model config
    n_layer = 4
    n_head = 4
    n_embd = 128
    block_size = 256
    batch_size = args.batch_size

    # Build nanoGPT command
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
        f"--batch_size={batch_size}",
        f"--max_iters={args.max_iters}",
        f"--eval_interval=500",
        f"--log_interval=10",
        f"--eval_iters=200",
        f"--learning_rate={lr}",
        f"--optimizer={args.optimizer}",
        f"--muon_lr={lr}",
        f"--spectral_log_every=500",
        f"--spectral_full_svd=False",
        f"--wandb_log=False",
        f"--compile=False",
        f"--decay_lr=False",
        f"--gradient_accumulation_steps=1",
        f"--warmup_iters=100",
    ]

    meta = {
        "experiment": "01_muon_vs_adamw",
        "optimizer": args.optimizer,
        "lr": lr,
        "seed": args.seed,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "batch_size": batch_size,
        "max_iters": args.max_iters,
    }

    print(f"[01] Running {args.optimizer} | lr={lr} | seed={args.seed}")
    print(f"[01] Output: {out_dir}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.join(PROJECT_ROOT, "nanoGPT"))
    elapsed = time.time() - t0

    meta["elapsed_s"] = round(elapsed, 1)
    meta["returncode"] = result.returncode

    # Save stdout/stderr
    with open(os.path.join(out_dir, "stdout.txt"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(out_dir, "stderr.txt"), "w") as f:
        f.write(result.stderr)

    # Parse loss from stdout (nanoGPT prints "iter N: loss X.XXXX, ...")
    train_losses = []
    val_losses = []
    for line in result.stdout.splitlines():
        if line.startswith("iter ") and "loss" in line:
            parts = line.split()
            try:
                it = int(parts[1].rstrip(":"))
                loss_val = float(parts[3].rstrip(","))
                train_losses.append({"iter": it, "loss": loss_val})
            except (IndexError, ValueError):
                pass
        if "val loss" in line.lower():
            parts = line.split()
            for i, tok in enumerate(parts):
                if tok == "loss" and i + 1 < len(parts):
                    try:
                        vl = float(parts[i + 1].rstrip(","))
                        val_losses.append(vl)
                    except ValueError:
                        pass

    meta["train_losses"] = train_losses
    meta["val_losses"] = val_losses

    # Write summary JSON
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[01] Done in {elapsed:.1f}s. Summary -> {summary_path}")

    if result.returncode != 0:
        print("[01] WARNING: training returned non-zero exit code.")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    return meta


if __name__ == "__main__":
    args = parse_args()
    run(args)
