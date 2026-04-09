#!/usr/bin/env python3
"""
06_weight_decay_ablation.py -- Does weight decay destroy Muon's spectral advantage?
Sweep: --weight_decay {0, 0.01, 0.1, 0.3} for both optimizers.

Usage:
  python experiments/nanogpt/06_weight_decay_ablation.py \
      --weight_decay 0.1 --optimizer muon --seed 42

SLURM: one (weight_decay, optimizer, seed) triple per job.
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

WEIGHT_DECAYS = [0.0, 0.01, 0.1, 0.3]


def parse_args():
    p = argparse.ArgumentParser(description="Weight decay ablation")
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
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
    tag = f"{args.optimizer}_wd{args.weight_decay}_s{args.seed}"
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "results", "06_weight_decay_ablation", tag)
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
        f"--weight_decay={args.weight_decay}",
        f"--muon_optimizer={args.optimizer}",
        f"--muon_lr={lr}",
        f"--spectral_log_every=500",
        f"--spectral_full_svd=True",
        f"--wandb_log=False",
        f"--compile=False",
        f"--decay_lr=False",
        f"--gradient_accumulation_steps=1",
        f"--warmup_iters=100",
    ]

    meta = {
        "experiment": "06_weight_decay_ablation",
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "lr": lr,
        "seed": args.seed,
        "max_iters": args.max_iters,
    }

    print(f"[06] wd={args.weight_decay} | {args.optimizer} | seed={args.seed}")
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

    # Read spectral entropy from spectral log
    spectral_log = os.path.join(out_dir, "spectral", "spectral_log.jsonl")
    spectral_summary = []
    if os.path.isfile(spectral_log):
        with open(spectral_log) as f:
            for line in f:
                snap = json.loads(line)
                avg_entropy = sum(r["spectral_entropy"] for r in snap["layers"]) / max(len(snap["layers"]), 1)
                spectral_summary.append({
                    "iter": snap["iter"],
                    "avg_spectral_entropy": round(avg_entropy, 4),
                })
    meta["spectral_summary"] = spectral_summary

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[06] Done in {elapsed:.1f}s -> {summary_path}")
    return meta


if __name__ == "__main__":
    run(parse_args())
