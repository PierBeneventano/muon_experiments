#!/usr/bin/env python3
"""
train_vit_cifar.py -- Architecture-matched Vision Transformer on CIFAR-10.

Purpose: vision counterpart to the NanoGPT experiments so that spectral dynamics
(stable rank, spectral entropy, condition number) are directly comparable across
modalities under Muon vs AdamW. This is the core experiment for Section 7 of
the paper ("Why Language Specifically") -- we expect AdamW to collapse rank on
language but preserve it on vision, while Muon preserves rank on both.

Architecture matches the 4-layer 4-head d_model=128 NanoGPT exactly:
    n_layer=4, n_head=4, n_embd=128, image 32x32 -> 8x8=64 patches + cls token.
    Sequence length 65 (vs 256 for NanoGPT -- OK, still enough tokens to see rank).

We reuse MuonOptimizer, SpectralLogger, compute_spectral_metrics verbatim from
nanoGPT/muon_utils.py (model-agnostic). We write a non-causal SelfAttention
variant + patch embedding + cls classification head locally.

Usage:
  python experiments/vision/train_vit_cifar.py --optimizer muon --seed 42
  python experiments/vision/train_vit_cifar.py --optimizer adamw --seed 137

Output (per run): summary.json + spectral/spectral_log.jsonl in out_dir.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "nanoGPT"))

# Reused from nanoGPT -- model-agnostic
from muon_utils import MuonOptimizer, SpectralLogger  # noqa: E402


# =============================================================================
# Model: Vision Transformer matched to NanoGPT 4L/4H/128 config
# =============================================================================

class SelfAttention(nn.Module):
    """Non-causal multi-head self-attention. Matches CausalSelfAttention
    from nanoGPT/model.py line-for-line, except is_causal=False and no bias mask."""

    def __init__(self, n_embd, n_head, dropout=0.0, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Same as nanoGPT MLP."""
    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Non-causal transformer block matched to nanoGPT.Block."""
    def __init__(self, n_embd, n_head, dropout=0.0, bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = SelfAttention(n_embd, n_head, dropout=dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout=dropout, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer for CIFAR-10.
    Image 32x32x3 -> 8x8 patches of 4x4x3 = 64 patches + 1 cls = 65 tokens.
    Reuses Block structure from nanoGPT for a fair rank comparison."""

    def __init__(self, n_layer=4, n_head=4, n_embd=128, patch_size=4,
                 image_size=32, n_classes=10, in_channels=3, dropout=0.0, bias=True):
        super().__init__()
        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.n_embd = n_embd

        # Patch embedding via Conv2d (equivalent to a linear projection per patch)
        self.patch_embed = nn.Conv2d(in_channels, n_embd, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, n_embd))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout=dropout, bias=bias) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd, bias=bias)
        self.head = nn.Linear(n_embd, n_classes, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)              # (B, n_embd, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)     # (B, n_patches, n_embd)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)       # (B, n_patches+1, n_embd)
        x = self.dropout(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x[:, 0])  # logits on cls token


# =============================================================================
# CIFAR-10 data
# =============================================================================

def get_cifar_loaders(batch_size, data_root, num_workers=2):
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise RuntimeError("torchvision is required for CIFAR-10") from e

    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2470, 0.2435, 0.2616)

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    val_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# =============================================================================
# Training loop
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ViT on CIFAR-10 (Muon vs AdamW)")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=None, help="default 0.02 muon, 1e-3 adamw")
    p.add_argument("--momentum", type=float, default=0.95, help="Muon momentum / AdamW beta1")
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--spectral_log_every", type=int, default=100)
    p.add_argument("--spectral_full_svd", type=lambda x: str(x).lower() in ("true", "1"), default=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def default_lr(optimizer):
    return 0.02 if optimizer == "muon" else 1e-3


@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters):
    model.eval()
    losses, accs = [], []
    for i, (x, y) in enumerate(loader):
        if i >= eval_iters:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(F.cross_entropy(logits, y).item())
        accs.append((logits.argmax(-1) == y).float().mean().item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan"), \
           float(np.mean(accs)) if accs else float("nan")


def run(args):
    lr = args.lr if args.lr is not None else default_lr(args.optimizer)

    base_dir = args.output_dir or os.path.join(PROJECT_ROOT, "results", "vision", "lang_vs_vision")
    tag = f"vit_{args.optimizer}_s{args.seed}"
    out_dir = os.path.join(base_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    data_root = args.data_root or os.path.join(PROJECT_ROOT, "data", "cifar10")
    os.makedirs(data_root, exist_ok=True)

    meta = {
        "experiment": "vision_vit_cifar",
        "optimizer": args.optimizer,
        "seed": args.seed,
        "lr": lr,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "max_iters": args.max_iters,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
    }

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"[vit] optimizer={args.optimizer} seed={args.seed} lr={lr} device={device}", flush=True)
    print(f"[vit] out_dir={out_dir}", flush=True)
    print(f"[vit] n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} patch={args.patch_size}", flush=True)

    # Data
    print(f"[vit] loading CIFAR-10 from {data_root}...", flush=True)
    train_loader, val_loader = get_cifar_loaders(args.batch_size, data_root)
    print(f"[vit] train batches={len(train_loader)} val batches={len(val_loader)}", flush=True)

    # Model
    model = ViT(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                patch_size=args.patch_size, image_size=32, n_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[vit] params={n_params:,}", flush=True)
    meta["n_params"] = n_params

    # Optimizer (replicates nanoGPT/train.py:211-217 logic)
    if args.optimizer == "muon":
        optim = MuonOptimizer(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, momentum=args.momentum, weight_decay=args.weight_decay,
        )
        print(f"[vit] MuonOptimizer lr={lr} momentum={args.momentum} wd={args.weight_decay}", flush=True)
    else:
        optim = torch.optim.AdamW(
            model.parameters(), lr=lr,
            betas=(args.momentum, args.beta2), weight_decay=args.weight_decay,
        )
        print(f"[vit] AdamW lr={lr} betas=({args.momentum},{args.beta2}) wd={args.weight_decay}", flush=True)

    # Spectral logging (reused from nanoGPT)
    spectral_dir = os.path.join(out_dir, "spectral")
    spectral_logger = None
    if args.spectral_log_every > 0:
        spectral_logger = SpectralLogger(
            spectral_dir,
            log_every=args.spectral_log_every,
            full_svd=args.spectral_full_svd,
        )
        print(f"[vit] spectral logging every {args.spectral_log_every} iters (full_svd={args.spectral_full_svd})", flush=True)

    # Training loop -- iteration-based, mirrors NanoGPT's style
    model.train()
    train_iter = iter(train_loader)
    train_losses = []
    val_metrics = []

    t0 = time.time()
    for it in range(args.max_iters + 1):
        # Evaluate
        if it % args.eval_interval == 0 or it == args.max_iters:
            vl, va = estimate_loss(model, val_loader, device, args.eval_iters)
            tl, ta = estimate_loss(model, train_loader, device, args.eval_iters)
            print(f"[vit] iter {it}: train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
            val_metrics.append({"iter": it, "train_loss": tl, "train_acc": ta,
                                "val_loss": vl, "val_acc": va})

        if spectral_logger is not None:
            spectral_logger.maybe_log(model, it, extra={"train_loss": train_losses[-1]["loss"] if train_losses else None})

        if it == args.max_iters:
            break

        # Training step
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if it % args.log_interval == 0:
            train_losses.append({"iter": it, "loss": loss.item()})
            if it % (args.log_interval * 10) == 0:
                print(f"[vit] iter {it}: loss={loss.item():.4f}", flush=True)

    elapsed = time.time() - t0
    meta["elapsed_s"] = round(elapsed, 1)
    meta["train_losses"] = train_losses
    meta["val_metrics"] = val_metrics
    if val_metrics:
        meta["final_val_loss"] = val_metrics[-1]["val_loss"]
        meta["final_val_acc"] = val_metrics[-1]["val_acc"]
        meta["best_val_acc"] = max(m["val_acc"] for m in val_metrics)
        meta["best_val_loss"] = min(m["val_loss"] for m in val_metrics)

    # Count spectral snapshots
    n_snapshots = 0
    spectral_log = os.path.join(spectral_dir, "spectral_log.jsonl")
    if os.path.isfile(spectral_log):
        with open(spectral_log) as f:
            n_snapshots = sum(1 for _ in f)
    meta["n_spectral_snapshots"] = n_snapshots

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[vit] DONE in {elapsed:.1f}s -> {summary_path} ({n_snapshots} spectral snapshots)", flush=True)


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        import traceback
        print("[vit] UNCAUGHT EXCEPTION:", flush=True)
        traceback.print_exc()
        sys.exit(1)
