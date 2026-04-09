#!/bin/bash
# all_jobs.sh -- Submit all NanoGPT Muon experiments as SLURM jobs.
#
# Usage:
#   bash experiments/nanogpt/all_jobs.sh          # submit everything
#   bash experiments/nanogpt/all_jobs.sh --dry-run # print commands without submitting
#
# Prerequisites:
#   1. Run setup.sh first to clone nanoGPT and apply the Muon patch.
#   2. Adjust SLURM_PARTITION, SLURM_ACCOUNT, etc. below for your cluster.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="$PROJECT_ROOT/results"

# ---------------------------------------------------------------------------
# SLURM configuration -- edit for your cluster
# ---------------------------------------------------------------------------
SLURM_PARTITION="${SLURM_PARTITION:-mit_normal_gpu}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_TIME_SHORT="02:00:00"       # ~2 min experiments, generous wall-time
SLURM_TIME_MEDIUM="04:00:00"      # ~10 min experiments
SLURM_TIME_LONG="06:00:00"        # LR sweep (many short runs, 6h max on mit_normal_gpu)
CONDA_ENV="${CONDA_ENV:-muon}"     # conda environment name
SLURM_GPUS=1
SLURM_CPUS=4
SLURM_MEM="32G"

DRY_RUN=0
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "=== DRY RUN -- commands will be printed but not submitted ==="
fi

# Seeds for replication
SEEDS=(42 137 2024)

# ---------------------------------------------------------------------------
# Helper: submit a single SLURM job
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

source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniforge 2>/dev/null || true
conda activate $CONDA_ENV 2>/dev/null || true

cd "$PROJECT_ROOT"
$CMD
EOF
    echo "Submitted: $JOB_NAME"
}

# ===================================================================
# Experiment 01: Muon vs AdamW
# ===================================================================
echo "--- Experiment 01: Muon vs AdamW ---"
for OPT in muon adamw; do
    for SEED in "${SEEDS[@]}"; do
        submit "exp01_${OPT}_s${SEED}" "$SLURM_TIME_SHORT" \
            python experiments/nanogpt/01_muon_vs_adamw.py \
                --output_dir "$RESULTS_ROOT/nanogpt/01_muon_vs_adamw" \
                --muon_optimizer "$OPT" --seed "$SEED"
    done
done

# ===================================================================
# Experiment 02: Batch-size sweep
# ===================================================================
echo "--- Experiment 02: Batch-size sweep ---"
for BS in 8 16 32 64 128 256 512; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            submit "exp02_${OPT}_bs${BS}_s${SEED}" "$SLURM_TIME_SHORT" \
                python experiments/nanogpt/02_batch_size_sweep.py \
                --output_dir "$RESULTS_ROOT/nanogpt/02_batch_size_sweep" \
                    --batch_size "$BS" --muon_optimizer "$OPT" --seed "$SEED"
        done
    done
done

# ===================================================================
# Experiment 03: LR sweep (large grid)
# ===================================================================
echo "--- Experiment 03: LR sweep ---"
for LR in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1; do
    for BS in 8 16 32 64 128 256 512; do
        for OPT in muon adamw; do
            submit "exp03_${OPT}_lr${LR}_bs${BS}" "$SLURM_TIME_SHORT" \
                python experiments/nanogpt/03_lr_sweep.py \
                --output_dir "$RESULTS_ROOT/nanogpt/03_lr_sweep" \
                    --lr "$LR" --batch_size "$BS" --muon_optimizer "$OPT" --seed 42
        done
    done
done

# ===================================================================
# Experiment 04: Spectral tracking
# ===================================================================
echo "--- Experiment 04: Spectral tracking ---"
for OPT in muon adamw; do
    for SEED in "${SEEDS[@]}"; do
        submit "exp04_${OPT}_s${SEED}" "$SLURM_TIME_SHORT" \
            python experiments/nanogpt/04_spectral_tracking.py \
                --output_dir "$RESULTS_ROOT/nanogpt/04_spectral_tracking" \
                --muon_optimizer "$OPT" --seed "$SEED" --log_every 100
    done
done

# ===================================================================
# Experiment 05: Feature acquisition
# ===================================================================
echo "--- Experiment 05: Feature acquisition ---"
for OPT in muon adamw; do
    for SEED in "${SEEDS[@]}"; do
        submit "exp05_${OPT}_s${SEED}" "$SLURM_TIME_SHORT" \
            python experiments/nanogpt/05_feature_acquisition.py \
                --output_dir "$RESULTS_ROOT/nanogpt/05_feature_acquisition" \
                --muon_optimizer "$OPT" --seed "$SEED" --log_every 100 --top_k 10
    done
done

# ===================================================================
# Experiment 06: Weight decay ablation
# ===================================================================
echo "--- Experiment 06: Weight decay ablation ---"
for WD in 0 0.01 0.1 0.3; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            submit "exp06_${OPT}_wd${WD}_s${SEED}" "$SLURM_TIME_SHORT" \
                python experiments/nanogpt/06_weight_decay_ablation.py \
                --output_dir "$RESULTS_ROOT/nanogpt/06_weight_decay_ablation" \
                    --weight_decay "$WD" --muon_optimizer "$OPT" --seed "$SEED"
        done
    done
done

# ===================================================================
# Experiment 07: Momentum ablation (Muon only)
# ===================================================================
echo "--- Experiment 07: Momentum ablation ---"
for MOM in 0 0.5 0.9 0.95 0.99; do
    for SEED in "${SEEDS[@]}"; do
        submit "exp07_mom${MOM}_s${SEED}" "$SLURM_TIME_SHORT" \
            python experiments/nanogpt/07_momentum_ablation.py \
                --output_dir "$RESULTS_ROOT/nanogpt/07_momentum_ablation" \
                --momentum "$MOM" --seed "$SEED"
    done
done

# ===================================================================
# Experiment 08: Model scale (width)
# ===================================================================
echo "--- Experiment 08: Model scale ---"
for EMBD in 64 128 256 512; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            submit "exp08_${OPT}_embd${EMBD}_s${SEED}" "$SLURM_TIME_MEDIUM" \
                python experiments/nanogpt/08_model_scale.py \
                --output_dir "$RESULTS_ROOT/nanogpt/08_model_scale" \
                    --n_embd "$EMBD" --muon_optimizer "$OPT" --seed "$SEED"
        done
    done
done

# ===================================================================
# Experiment 09: Depth ablation
# ===================================================================
echo "--- Experiment 09: Depth ablation ---"
for DEPTH in 2 4 6 8; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            submit "exp09_${OPT}_depth${DEPTH}_s${SEED}" "$SLURM_TIME_MEDIUM" \
                python experiments/nanogpt/09_depth_ablation.py \
                --output_dir "$RESULTS_ROOT/nanogpt/09_depth_ablation" \
                    --n_layer "$DEPTH" --muon_optimizer "$OPT" --seed "$SEED"
        done
    done
done

# ===================================================================
# Experiment 10: Head ablation
# ===================================================================
echo "--- Experiment 10: Head ablation ---"
for HEADS in 1 2 4 8; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            submit "exp10_${OPT}_heads${HEADS}_s${SEED}" "$SLURM_TIME_SHORT" \
                python experiments/nanogpt/10_head_ablation.py \
                --output_dir "$RESULTS_ROOT/nanogpt/10_head_ablation" \
                    --n_head "$HEADS" --muon_optimizer "$OPT" --seed "$SEED"
        done
    done
done

# ===================================================================
# Experiment 11: S(mu) measurement (post-hoc, depends on 01)
# Submitted as held jobs -- release after experiment 01 completes.
# ===================================================================
echo "--- Experiment 11: S(mu) measurement (queued) ---"
echo "NOTE: Experiment 11 depends on checkpoints from Experiment 01."
echo "      Run it after Experiment 01 completes:"
echo "      python experiments/nanogpt/11_s_mu_measurement.py --checkpoint_dir results/01_muon_vs_adamw/<run>"

# ===================================================================
# Experiment 12: Regression vs classification
# ===================================================================
echo "--- Experiment 12: Regression vs classification ---"
for TASK in generation classification; do
    for OPT in muon adamw; do
        for SEED in "${SEEDS[@]}"; do
            submit "exp12_${TASK}_${OPT}_s${SEED}" "$SLURM_TIME_SHORT" \
                python experiments/nanogpt/12_regression_vs_cls.py \
                --output_dir "$RESULTS_ROOT/nanogpt/12_regression_vs_cls" \
                    --task "$TASK" --muon_optimizer "$OPT" --seed "$SEED"
        done
    done
done

# ===================================================================
# Summary
# ===================================================================
echo ""
echo "=== Job submission complete ==="
echo "Total jobs (approximate):"
echo "  01: 6 jobs    (2 opts x 3 seeds)"
echo "  02: 42 jobs   (7 BS x 2 opts x 3 seeds)"
echo "  03: 98 jobs   (7 LR x 7 BS x 2 opts, seed=42 only)"
echo "  04: 6 jobs    (2 opts x 3 seeds)"
echo "  05: 6 jobs    (2 opts x 3 seeds)"
echo "  06: 24 jobs   (4 WD x 2 opts x 3 seeds)"
echo "  07: 15 jobs   (5 momenta x 3 seeds, muon only)"
echo "  08: 24 jobs   (4 widths x 2 opts x 3 seeds)"
echo "  09: 24 jobs   (4 depths x 2 opts x 3 seeds)"
echo "  10: 24 jobs   (4 heads x 2 opts x 3 seeds)"
echo "  11: post-hoc  (manual after 01 completes)"
echo "  12: 12 jobs   (2 tasks x 2 opts x 3 seeds)"
echo "  ---------"
echo "  Total: ~281 SLURM jobs"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in:   $RESULTS_ROOT/"
