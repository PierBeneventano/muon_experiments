#!/usr/bin/env python3
"""
patch_muon.py -- Patch nanoGPT's train.py with:
  1. MuonOptimizer class (polar-decomposition step via torch.linalg.svd)
  2. --optimizer CLI flag ('adamw' | 'muon')
  3. Spectral logging hooks (SVD, spectral entropy, S(mu))
  4. Checkpointing of spectral logs

Run from the project root (parent of nanoGPT/):
    python experiments/nanogpt/patch_muon.py
"""

import os
import sys
import textwrap
import re

# ---------------------------------------------------------------------------
# Locate nanoGPT/train.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TRAIN_PY = os.path.join(PROJECT_ROOT, "nanoGPT", "train.py")

if not os.path.isfile(TRAIN_PY):
    sys.exit(f"ERROR: Cannot find {TRAIN_PY}. Run setup.sh first to clone nanoGPT.")

# ---------------------------------------------------------------------------
# Backup original
# ---------------------------------------------------------------------------
BACKUP = TRAIN_PY + ".orig"
if not os.path.isfile(BACKUP):
    with open(TRAIN_PY, "r") as f:
        orig = f.read()
    with open(BACKUP, "w") as f:
        f.write(orig)
    print(f"Backed up original train.py -> {BACKUP}")
else:
    with open(BACKUP, "r") as f:
        orig = f.read()
    print(f"Using existing backup {BACKUP}")

# ---------------------------------------------------------------------------
# The Muon optimizer + spectral-logging code we inject
# ---------------------------------------------------------------------------

MUON_BLOCK = textwrap.dedent(r'''
# ============================================================================
# BEGIN MUON PATCH -- injected by experiments/nanogpt/patch_muon.py
# ============================================================================
import json, math, time as _time, pathlib

# ---- Muon Optimizer --------------------------------------------------------

class MuonOptimizer(torch.optim.Optimizer):
    """
    Muon: momentum + polar decomposition update for weight matrices.
    Biases and 1-D parameters fall back to plain SGD.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dim() < 2:
                    # 1-D params (biases, layernorm): plain SGD
                    p.data.add_(grad, alpha=-group['lr'])
                    continue
                # Decoupled weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                # Momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                # Polar decomposition via SVD
                U, S, Vt = torch.linalg.svd(buf, full_matrices=False)
                polar = U @ Vt
                p.data.add_(polar, alpha=-group['lr'])
        return loss

# ---- Spectral logging utilities --------------------------------------------

def _spectral_entropy(singular_values):
    """Unnormalised spectral entropy H(s) = -sum p_k ln p_k."""
    s = singular_values.clamp(min=1e-12)
    p = s / s.sum()
    return -(p * p.log()).sum().item()

def _spectral_entropy_normalized(singular_values):
    """Normalised spectral entropy H(s) / H_max in [0, 1]."""
    h = _spectral_entropy(singular_values)
    h_max = math.log(len(singular_values)) if len(singular_values) > 1 else 1.0
    return h / h_max if h_max > 0 else 0.0

def _s_mu_functional(singular_values):
    """S(mu) = sum_{i!=j} 1/(s_i + s_j)^2 — the spectral functional from the paper."""
    s = singular_values.clamp(min=1e-12).cpu().numpy()
    n = len(s)
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += 1.0 / (s[i] + s[j])**2
    return total

def _effective_rank(singular_values):
    """exp(H(s)) — continuous rank measure in [1, min(m,n)]."""
    h = _spectral_entropy(singular_values)
    return math.exp(h)

def _condition_number(singular_values):
    """sigma_1 / sigma_min."""
    s = singular_values.clamp(min=1e-12)
    return (s[0] / s[-1]).item()

def _stable_rank(singular_values):
    """||W||_F^2 / ||W||_op^2 = sum(s^2) / s_1^2."""
    s = singular_values.clamp(min=1e-12)
    return ((s**2).sum() / s[0]**2).item()

def _nuclear_to_frobenius(singular_values):
    """||W||_* / ||W||_F = sum(s) / sqrt(sum(s^2))."""
    s = singular_values.clamp(min=1e-12)
    return (s.sum() / (s**2).sum().sqrt()).item()

def _gini_coefficient(singular_values):
    """Gini coefficient of singular values — measures spectral inequality."""
    s = singular_values.clamp(min=1e-12).cpu().numpy()
    s = np.sort(s)
    n = len(s)
    idx = np.arange(1, n + 1)
    return (2.0 * (idx * s).sum() / (n * s.sum()) - (n + 1) / n)

import numpy as np  # for gini

def compute_spectral_metrics(model, full_svd=False):
    """
    Compute comprehensive spectral metrics for every 2-D parameter.
    Logs everything we could need so we never have to rerun.
    """
    records = []
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        with torch.no_grad():
            S = torch.linalg.svdvals(param.data.float())
        rec = dict(
            name=name,
            shape=list(param.shape),
            # Entropy metrics
            spectral_entropy=_spectral_entropy(S),
            spectral_entropy_normalized=_spectral_entropy_normalized(S),
            h_max=math.log(min(param.shape[0], param.shape[1])),
            # Rank and condition
            effective_rank=_effective_rank(S),
            stable_rank=_stable_rank(S),
            condition_number=_condition_number(S),
            numerical_rank=int((S > 1e-6 * S[0]).sum().item()),
            # Norms and ratios
            frobenius_norm=(S**2).sum().sqrt().item(),
            nuclear_norm=S.sum().item(),
            operator_norm=S[0].item(),
            nuclear_to_frobenius=_nuclear_to_frobenius(S),
            # Spectral shape
            top1_sv=S[0].item(),
            top5_sv=S[:5].cpu().tolist() if len(S) >= 5 else S.cpu().tolist(),
            bottom5_sv=S[-5:].cpu().tolist() if len(S) >= 5 else S.cpu().tolist(),
            sv_mean=S.mean().item(),
            sv_std=S.std().item(),
            gini=_gini_coefficient(S),
            # S(mu) functional (expensive for large matrices, skip if dim > 512)
            s_mu=(_s_mu_functional(S) if len(S) <= 512 else None),
        )
        if full_svd:
            rec['singular_values'] = S.cpu().tolist()
        records.append(rec)
    return records


class SpectralLogger:
    """Accumulates spectral snapshots and writes them to a JSONL file."""

    def __init__(self, out_dir, log_every=500, full_svd=False):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.full_svd = full_svd
        self.log_path = self.out_dir / "spectral_log.jsonl"
        self.buffer = []

    def maybe_log(self, model, iter_num, extra=None):
        if iter_num % self.log_every != 0:
            return
        metrics = compute_spectral_metrics(model, full_svd=self.full_svd)
        record = dict(iter=iter_num, layers=metrics)
        if extra:
            record.update(extra)
        self.buffer.append(record)
        # append to disk immediately so nothing is lost on crash
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def state_dict(self):
        return dict(buffer=self.buffer, log_every=self.log_every,
                    full_svd=self.full_svd)

    def load_state_dict(self, sd):
        self.buffer = sd.get('buffer', [])
        self.log_every = sd.get('log_every', self.log_every)
        self.full_svd = sd.get('full_svd', self.full_svd)

# ---- New config knobs ------------------------------------------------------
# These can be overridden from the command line just like any other nanoGPT
# config variable, e.g.  python train.py --optimizer=muon --spectral_log_every=100
optimizer = 'adamw'             # 'adamw' | 'muon'
muon_lr = 0.02
muon_momentum = 0.95
spectral_log_every = 500        # 0 = disabled
spectral_full_svd = True        # save full singular-value vectors (small for n_embd<=512)
seed = 1337                     # random seed (nanoGPT has no built-in seed config)

# ============================================================================
# END MUON PATCH
# ============================================================================
''')

# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def inject_after_imports(src: str, block: str) -> str:
    """Insert *block* right after the last top-level import / from-import."""
    lines = src.split('\n')
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_idx = i
    # skip any blank lines immediately after
    insert_at = last_import_idx + 1
    while insert_at < len(lines) and lines[insert_at].strip() == '':
        insert_at += 1
    lines.insert(insert_at, block)
    return '\n'.join(lines)


def patch_optimizer_creation(src: str) -> str:
    """
    Replace the optimizer construction section so it honours --optimizer.
    nanoGPT builds its optimizer via model.configure_optimizers(...).
    We wrap that call.
    """
    # Find the line that calls configure_optimizers
    marker = "model.configure_optimizers"
    if marker not in src:
        print("WARNING: Could not find model.configure_optimizers in train.py. "
              "Optimizer patching skipped.")
        return src

    # We replace the optimizer creation block with one that branches on
    # the `optimizer` config variable.
    # The original block typically looks like:
    #   optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    # We need to keep the original for adamw and add muon branch.

    replacement = textwrap.dedent('''\
    # --- MUON PATCH: optimizer selection ---
    if optimizer == 'muon':
        _muon_params = [p for p in model.parameters() if p.requires_grad]
        _optimizer_obj = MuonOptimizer(
            _muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=weight_decay,
        )
        print(f"Using MuonOptimizer (lr={muon_lr}, momentum={muon_momentum}, wd={weight_decay})")
    else:
        _optimizer_obj = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        print(f"Using AdamW (lr={learning_rate}, wd={weight_decay})")
    # Alias so the rest of the code sees `optimizer` as the object, but we
    # need the *string* config value preserved.  nanoGPT re-uses the name
    # `optimizer` for the object, so we stash the string first.
    _optimizer_choice = optimizer  # 'adamw' or 'muon'
    optimizer = _optimizer_obj
    # --- END MUON PATCH: optimizer selection ---''')

    # Use regex to find the full configure_optimizers call and replace it
    pattern = re.compile(
        r'^(\s*)optimizer\s*=\s*model\.configure_optimizers\(.*?\).*$',
        re.MULTILINE
    )
    match = pattern.search(src)
    if not match:
        print("WARNING: regex for configure_optimizers failed. Trying simpler match.")
        # Fallback: line-level replacement
        lines = src.split('\n')
        new_lines = []
        for line in lines:
            if 'model.configure_optimizers' in line and 'optimizer' in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(textwrap.indent(replacement, indent))
            else:
                new_lines.append(line)
        return '\n'.join(new_lines)

    indent = match.group(1)
    src = src[:match.start()] + textwrap.indent(replacement, indent) + src[match.end():]
    return src


def patch_training_loop(src: str) -> str:
    """
    Insert spectral-logging calls into the training loop and checkpoint
    saving logic.
    """
    # --- 1. Initialise SpectralLogger before the loop ---
    loop_marker = "for iter_num in range"  # nanoGPT's main loop
    alt_marker = "while True:"             # some versions use while True
    marker_used = None
    for m in [loop_marker, alt_marker]:
        if m in src:
            marker_used = m
            break
    if marker_used is None:
        print("WARNING: Could not locate main training loop. Spectral logging skipped.")
        return src

    init_snippet = textwrap.dedent('''\
    # --- MUON PATCH: seed + spectral logger init ---
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    import numpy as _np; _np.random.seed(seed)
    _spectral_logger = None
    if spectral_log_every > 0:
        _spectral_out = os.path.join(out_dir, 'spectral')
        _spectral_logger = SpectralLogger(
            _spectral_out, log_every=spectral_log_every, full_svd=spectral_full_svd
        )
    # --- END MUON PATCH: seed + spectral logger init ---
    ''')

    idx = src.index(marker_used)
    # Walk backwards to find a good insertion point (blank line before the loop)
    insert_at = idx
    while insert_at > 0 and src[insert_at - 1] != '\n':
        insert_at -= 1
    src = src[:insert_at] + init_snippet + "\n" + src[insert_at:]

    # --- 2. Log after each iteration ---
    # We look for the loss.backward() call or the optimizer.step() call
    # and insert our logging right after scaler.update() or optimizer.step()
    step_patterns = [
        'scaler.update()',
        'optimizer.step()',
    ]
    log_call = textwrap.dedent('''\
            # --- MUON PATCH: spectral log ---
            if _spectral_logger is not None:
                _extra = {'train_loss': lossf if 'lossf' in dir() else None}
                # Log gradient norms per layer (cheap, always available)
                try:
                    _grad_norms = {}
                    for _pname, _pparam in model.named_parameters():
                        if _pparam.grad is not None and _pparam.dim() >= 2:
                            _grad_norms[_pname] = _pparam.grad.data.norm().item()
                    _extra['grad_norms'] = _grad_norms
                except Exception:
                    pass
                _spectral_logger.maybe_log(model, iter_num, extra=_extra)
            # --- END MUON PATCH ---
    ''')

    for pat in step_patterns:
        if pat in src:
            # Insert after the first occurrence
            pos = src.index(pat) + len(pat)
            # skip to end of line
            while pos < len(src) and src[pos] != '\n':
                pos += 1
            src = src[:pos + 1] + log_call + src[pos + 1:]
            break

    # --- 3. Include spectral logger state in checkpoints ---
    # nanoGPT saves checkpoints with a dict that typically includes
    # 'model', 'optimizer', 'model_args', 'iter_num', 'best_val_loss', 'config'
    ckpt_marker = "'config': config,"
    if ckpt_marker in src:
        extra = "            'spectral_logger': _spectral_logger.state_dict() if _spectral_logger else None,\n"
        pos = src.index(ckpt_marker) + len(ckpt_marker)
        while pos < len(src) and src[pos] != '\n':
            pos += 1
        src = src[:pos + 1] + extra + src[pos + 1:]

    return src


def patch_resume_checkpoint(src: str) -> str:
    """Restore spectral logger state when resuming from a checkpoint."""
    resume_marker = "checkpoint = torch.load"
    if resume_marker not in src:
        return src
    # Find the block where state is restored
    restore_marker = "optimizer.load_state_dict"
    if restore_marker not in src:
        return src
    pos = src.index(restore_marker)
    # go to end of that line
    while pos < len(src) and src[pos] != '\n':
        pos += 1
    snippet = textwrap.dedent('''\
        # --- MUON PATCH: restore spectral logger ---
        if _spectral_logger is not None and checkpoint.get('spectral_logger'):
            _spectral_logger.load_state_dict(checkpoint['spectral_logger'])
        # --- END MUON PATCH ---
    ''')
    # We'll inject this later in the file after the training loop init,
    # so it's safe even if _spectral_logger doesn't exist yet at resume time.
    # Actually, for safety, wrap in try/except.
    safe_snippet = textwrap.dedent('''\
        # --- MUON PATCH: restore spectral logger ---
        try:
            if '_spectral_logger' in dir() and _spectral_logger is not None:
                _sl_state = checkpoint.get('spectral_logger')
                if _sl_state:
                    _spectral_logger.load_state_dict(_sl_state)
        except Exception:
            pass
        # --- END MUON PATCH ---
    ''')
    src = src[:pos + 1] + safe_snippet + src[pos + 1:]
    return src


# ---------------------------------------------------------------------------
# Apply all patches
# ---------------------------------------------------------------------------
def main():
    src = orig

    # 1. Inject Muon class + utilities after imports
    src = inject_after_imports(src, MUON_BLOCK)

    # 2. Patch optimizer creation
    src = patch_optimizer_creation(src)

    # 3. Patch training loop for spectral logging
    src = patch_training_loop(src)

    # 4. Patch checkpoint resume
    src = patch_resume_checkpoint(src)

    # Write patched file
    with open(TRAIN_PY, 'w') as f:
        f.write(src)

    print(f"Patched {TRAIN_PY} successfully.")
    print("New config variables available:")
    print("  optimizer          = 'adamw' | 'muon'")
    print("  muon_lr            = 0.02")
    print("  muon_momentum      = 0.95")
    print("  spectral_log_every = 500  (0 to disable)")
    print("  spectral_full_svd  = False")


if __name__ == '__main__':
    main()
