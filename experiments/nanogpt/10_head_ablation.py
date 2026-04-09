#!/usr/bin/env python3
"""
10_head_ablation.py -- Does attention head count affect spectral properties?
Sweep: --n_head {1, 2, 4, 8}  (n_embd=128, so all divide evenly)

Usage:
  python experiments/nanogpt/10_head_ablation.py \
      --n_head 4 --optimizer muon --seed 42

SLURM: one (n_head, optimizer, seed) triple per job.
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

HEADS = [1, 2, 4, 8]


def parse_args():
    p = argparse.ArgumentParser(description="Attention head count ablation")
    p.add_argument("--n_head", type=int, required=True, choices=HEADS)
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
    tag = f"{args.optimizer}_heads{args.n_head}_s{args.seed}"
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "results", "10_head_ablation", tag)
    os.makedirs(out_dir, exist_ok=True)

    n_layer, n_embd, block_size = 4, 128, 256

    cmd = [
        sys.executable, TRAIN_PY,
        f"--out_dir={out_dir}",
        f"--seed={args.seed}",
        f"--device={args.device}",
        f"--dataset=shakespeare_char",
        f"--n_layer={n_layer}",
        f"--n_head={args.n_head}",
        f"--n_embd={n_embd}",
        f"--block_size={block_size}",
        f"--batch_size=64",
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
        "experiment": "10_head_ablation",
        "optimizer": args.optimizer,
        "n_head": args.n_head,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "lr": lr,
        "seed": args.seed,
        "max_iters": args.max_iters,
    }

    print(f"[10] n_head={args.n_head} | {args.optimizer} | seed={args.seed}")
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

    # Spectral
    spectral_log = os.path.join(out_dir, "spectral", "spectral_log.jsonl")
    spectral_summary = []
    if os.path.isfile(spectral_log):
        with open(spectral_log) as f:
            for line in f:
                snap = json.loads(line)
                # Focus on attention weight matrices
                attn_entropies = [r["spectral_entropy"] for r in snap["layers"]
                                  if "attn" in r["name"]]
                mlp_entropies = [r["spectral_entropy"] for r in snap["layers"]
                                 if "mlp" in r["name"] or "c_fc" in r["name"] or "c_proj" in r["name"]]
                spectral_summary.append({
                    "iter": snap["iter"],
                    "avg_attn_entropy": round(sum(attn_entropies) / max(len(attn_entropies), 1), 4),
                    "avg_mlp_entropy": round(sum(mlp_entropies) / max(len(mlp_entropies), 1), 4),
                })
    meta["spectral_summary"] = spectral_summary

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[10] Done in {elapsed:.1f}s -> {summary_path}")
    return meta


if __name__ == "__main__":
    run(parse_args())
