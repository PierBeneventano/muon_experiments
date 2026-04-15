#!/bin/bash
# resubmit_failed.sh -- Resubmit ONLY the failed NanoGPT experiments.
#
# Based on cluster results as of 2026-04-09:
#   exp01: 6/6  DONE
#   exp02: 42/42 DONE
#   exp03: 97/98 DONE (close enough)
#   exp04: 3/6  -- muon seeds failed (adamw done)
#   exp05: 0/6  -- all failed
#   exp06: 1/24 -- almost all failed
#   exp07: 0/15 -- all failed
#   exp08: 0/24 -- all failed
#   exp09: 0/24 -- never created
#   exp10: 0/24 -- all failed
#   exp12: 1/12 -- almost all failed
#
# Fix: uses eval "$(conda shell.bash hook)" for proper conda init in sbatch.
#
# Usage:
#   bash experiments/nanogpt/resubmit_failed.sh          # submit
#   bash experiments/nanogpt/resubmit_failed.sh --dry-run # print only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="$PROJECT_ROOT/results"

# ---------------------------------------------------------------------------
# SLURM configuration
# ---------------------------------------------------------------------------
SLURM_PARTITION="${SLURM_PARTITION:-ou_bcs_normal}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_TIME_SHORT="02:00:00"
SLURM_TIME_MEDIUM="04:00:00"
CONDA_ENV="${CONDA_ENV:-muon}"
SLURM_GPUS=1
SLURM_CPUS=4
SLURM_MEM="32G"

DRY_RUN=0
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "=== DRY RUN -- commands will be printed but not submitted ==="
fi

SEEDS=(42 137 2024)
TOTAL=0

# ---------------------------------------------------------------------------
# Helper: submit a single SLURM job (with fixed conda activation)
# ---------------------------------------------------------------------------
submit() {
    local JOB_NAME="$1"
    local TIME="$2"
    shift 2
    local CMD="$*"

    local ACCT_FLAG=""
    if [ -n "$SLURM_ACCOUNT" ]; then
        ACCT_FLAG="#SBATCH --account=$SLURM_ACCOUNT"
    fi

    local LOG_DIR="$RESULTS_ROOT/slurm_logs"
    mkdir -p "$LOG_DIR"

    TOTAL=$((TOTAL + 1))

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY] $JOB_NAME: $CMD"
        return
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$SLURM_PARTITION
$ACCT_FLAG
#SBATCH --gres=gpu:$SLURM_GPUS
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.err

# === BEGIN DIAGNOSTIC PREAMBLE ===
set -eo pipefail                       # fail fast + preserve pipe status
exec 2>&1                              # mirror stderr to stdout so the SLURM .out has everything
echo "[sbatch] host=\$(hostname) pid=\$\$ date=\$(date -Iseconds)"
echo "[sbatch] SLURM_JOB_ID=\${SLURM_JOB_ID:-n/a} partition=\${SLURM_JOB_PARTITION:-n/a}"
echo "[sbatch] node=\$(scontrol show hostname \${SLURM_JOB_NODELIST:-\$(hostname)} 2>/dev/null | head -1)"
echo "[sbatch] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -5 || echo "[sbatch] nvidia-smi not available"
# === END DIAGNOSTIC PREAMBLE ===

# Robust conda activation for non-interactive sbatch shells
source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniforge 2>/dev/null || true
eval "\$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
conda activate $CONDA_ENV
echo "[sbatch] conda activated: CONDA_PREFIX=\${CONDA_PREFIX:-unset}"

# Verify python is available
which python || { echo "ERROR: python not found after conda activate"; exit 1; }
python --version
python -c "import torch; print('[sbatch] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" 2>&1

cd "$PROJECT_ROOT"
echo "[sbatch] cwd=\$(pwd)"
echo "[sbatch] CMD=$CMD"

# Force unbuffered python (-u) so stdout/stderr flush on every line.
# \$CMD starts with 'python ...'; rewrite to 'python -u ...' for live logs.
CMD_UNBUFFERED=\$(echo "$CMD" | sed -E 's|^(\s*)python |\1python -u |')
echo "[sbatch] CMD_UNBUFFERED=\$CMD_UNBUFFERED"

# Run the command; on failure, echo a marker so the tail of the log makes the exit obvious
eval "\$CMD_UNBUFFERED"
RC=\$?
echo "[sbatch] exit_code=\$RC"
exit \$RC
EOF
    echo "Submitted: $JOB_NAME"
}

# Helper: check if a result already exists
has_result() {
    local DIR="$1"
    [ -f "$DIR/summary.json" ]
}

# ===================================================================
# Experiment 04: Spectral tracking (muon seeds failed)
# ===================================================================
echo "--- Experiment 04: Spectral tracking (muon only) ---"
for SEED in "${SEEDS[@]}"; do
    TAG="muon_s${SEED}"
    OUTDIR="$RESULTS_ROOT/nanogpt/04_spectral_tracking/$TAG"
    if ! has_result "$OUTDIR"; then
        submit "exp04_muon_s${SEED}" "$SLURM_TIME_SHORT" \
            python experiments/nanogpt/04_spectral_tracking.py \
                --output_dir "$RESULTS_ROOT/nanogpt/04_spectral_tracking" \
                --optimizer muon --seed "$SEED" --log_every 100
    else
        echo "  [SKIP] $TAG (already done)"
    fi
done

# ===================================================================
# Experiment 05: Feature acquisition (ALL failed)
# ===================================================================
echo "--- Experiment 05: Feature acquisition ---"
for OPT in muon adamw; do
    for SEED in "${SEEDS[@]}"; do
        TAG="${OPT}_s${SEED}"
        OUTDIR="$RESULTS_ROOT/nanogpt/05_feature_acquisition/$TAG"
        if ! has_result "$OUTDIR"; then
            submit "exp05_${OPT}_s${SEED}" "$SLURM_TIME_SHORT" \
                python experiments/nanogpt/05_feature_acquisition.py \
                    --output_dir "$RESULTS_ROOT/nanogpt/05_feature_acquisition" \
                    --optimizer "$OPT" --seed "$SEED" --log_every 100 --top_k 10
        else
            echo "  [SKIP] $TAG (already done)"
        fi
    done
done

# ===================================================================
# Experiment 06: Weight decay ablation (almost all failed)
# ===================================================================
echo "--- Experiment 06: Weight decay ablation ---"
for WD in 0 0.01 0.1 0.3; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            TAG="${OPT}_wd${WD}_s${SEED}"
            OUTDIR="$RESULTS_ROOT/nanogpt/06_weight_decay_ablation/$TAG"
            if ! has_result "$OUTDIR"; then
                submit "exp06_${OPT}_wd${WD}_s${SEED}" "$SLURM_TIME_SHORT" \
                    python experiments/nanogpt/06_weight_decay_ablation.py \
                    --output_dir "$RESULTS_ROOT/nanogpt/06_weight_decay_ablation" \
                        --weight_decay "$WD" --optimizer "$OPT" --seed "$SEED"
            else
                echo "  [SKIP] $TAG (already done)"
            fi
        done
    done
done

# ===================================================================
# Experiment 07: Momentum ablation (ALL failed)
# ===================================================================
echo "--- Experiment 07: Momentum ablation ---"
for MOM in 0 0.5 0.9 0.95 0.99; do
    for SEED in "${SEEDS[@]}"; do
        TAG="mom${MOM}_s${SEED}"
        OUTDIR="$RESULTS_ROOT/nanogpt/07_momentum_ablation/$TAG"
        if ! has_result "$OUTDIR"; then
            submit "exp07_mom${MOM}_s${SEED}" "$SLURM_TIME_SHORT" \
                python experiments/nanogpt/07_momentum_ablation.py \
                    --output_dir "$RESULTS_ROOT/nanogpt/07_momentum_ablation" \
                    --momentum "$MOM" --seed "$SEED"
        else
            echo "  [SKIP] $TAG (already done)"
        fi
    done
done

# ===================================================================
# Experiment 08: Model scale (ALL failed)
# ===================================================================
echo "--- Experiment 08: Model scale ---"
for EMBD in 64 128 256 512; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            TAG="${OPT}_embd${EMBD}_s${SEED}"
            OUTDIR="$RESULTS_ROOT/nanogpt/08_model_scale/$TAG"
            if ! has_result "$OUTDIR"; then
                submit "exp08_${OPT}_embd${EMBD}_s${SEED}" "$SLURM_TIME_MEDIUM" \
                    python experiments/nanogpt/08_model_scale.py \
                    --output_dir "$RESULTS_ROOT/nanogpt/08_model_scale" \
                        --n_embd "$EMBD" --optimizer "$OPT" --seed "$SEED"
            else
                echo "  [SKIP] $TAG (already done)"
            fi
        done
    done
done

# ===================================================================
# Experiment 09: Depth ablation (NEVER RAN — dir missing)
# ===================================================================
echo "--- Experiment 09: Depth ablation ---"
for DEPTH in 2 4 6 8; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            TAG="${OPT}_depth${DEPTH}_s${SEED}"
            OUTDIR="$RESULTS_ROOT/nanogpt/09_depth_ablation/$TAG"
            if ! has_result "$OUTDIR"; then
                submit "exp09_${OPT}_depth${DEPTH}_s${SEED}" "$SLURM_TIME_MEDIUM" \
                    python experiments/nanogpt/09_depth_ablation.py \
                    --output_dir "$RESULTS_ROOT/nanogpt/09_depth_ablation" \
                        --n_layer "$DEPTH" --optimizer "$OPT" --seed "$SEED"
            else
                echo "  [SKIP] $TAG (already done)"
            fi
        done
    done
done

# ===================================================================
# Experiment 10: Head ablation (ALL failed)
# ===================================================================
echo "--- Experiment 10: Head ablation ---"
for HEADS in 1 2 4 8; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            TAG="${OPT}_heads${HEADS}_s${SEED}"
            OUTDIR="$RESULTS_ROOT/nanogpt/10_head_ablation/$TAG"
            if ! has_result "$OUTDIR"; then
                submit "exp10_${OPT}_heads${HEADS}_s${SEED}" "$SLURM_TIME_SHORT" \
                    python experiments/nanogpt/10_head_ablation.py \
                    --output_dir "$RESULTS_ROOT/nanogpt/10_head_ablation" \
                        --n_head "$HEADS" --optimizer "$OPT" --seed "$SEED"
            else
                echo "  [SKIP] $TAG (already done)"
            fi
        done
    done
done

# ===================================================================
# Experiment 12: Regression vs classification (almost all failed)
# ===================================================================
echo "--- Experiment 12: Regression vs classification ---"
for TASK in generation classification; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            TAG="${TASK}_${OPT}_s${SEED}"
            OUTDIR="$RESULTS_ROOT/nanogpt/12_regression_vs_cls/$TAG"
            if ! has_result "$OUTDIR"; then
                submit "exp12_${TASK}_${OPT}_s${SEED}" "$SLURM_TIME_SHORT" \
                    python experiments/nanogpt/12_regression_vs_cls.py \
                    --output_dir "$RESULTS_ROOT/nanogpt/12_regression_vs_cls" \
                        --task "$TASK" --optimizer "$OPT" --seed "$SEED"
            else
                echo "  [SKIP] $TAG (already done)"
            fi
        done
    done
done

# ===================================================================
# Summary
# ===================================================================
echo ""
echo "=== Resubmission complete ==="
echo "Total jobs submitted: $TOTAL"
echo ""
echo "Expected breakdown:"
echo "  04: up to 3 jobs   (muon x 3 seeds)"
echo "  05: up to 6 jobs   (2 opts x 3 seeds)"
echo "  06: up to 23 jobs  (4 WD x 2 opts x 3 seeds, minus 1 done)"
echo "  07: up to 15 jobs  (5 momenta x 3 seeds)"
echo "  08: up to 24 jobs  (4 widths x 2 opts x 3 seeds)"
echo "  09: up to 24 jobs  (4 depths x 2 opts x 3 seeds)"
echo "  10: up to 24 jobs  (4 heads x 2 opts x 3 seeds)"
echo "  12: up to 11 jobs  (2 tasks x 2 opts x 3 seeds, minus 1 done)"
echo "  ---------"
echo "  Max: ~130 jobs (some may be skipped if results exist)"
echo ""
echo "Monitor with: squeue -u \$USER"
