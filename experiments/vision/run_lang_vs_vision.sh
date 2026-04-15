#!/bin/bash
# run_lang_vs_vision.sh -- Submit the language-vs-vision rank-collapse experiments.
#
# 6 jobs: {Muon, AdamW} x {seed 42, 137, 2024} on CIFAR-10 ViT.
# Matched to the NanoGPT exp04 spectral-tracking setup so rank curves are
# directly comparable across modalities.
#
# Shakespeare-side data is already collected via exp04 (3 AdamW seeds done;
# 3 Muon seeds being re-run in Phase A of the paper plan).
#
# Usage:
#   bash experiments/vision/run_lang_vs_vision.sh
#   bash experiments/vision/run_lang_vs_vision.sh --dry-run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="$PROJECT_ROOT/results"

SLURM_PARTITION="${SLURM_PARTITION:-ou_bcs_normal}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_TIME="02:00:00"
CONDA_ENV="${CONDA_ENV:-muon}"
SLURM_GPUS=1
SLURM_CPUS=4
SLURM_MEM="16G"

DRY_RUN=0
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "=== DRY RUN ==="
fi

SEEDS=(42 137 2024)
OPTIMIZERS=(muon adamw)
TOTAL=0

has_result() {
    local DIR="$1"
    [ -f "$DIR/summary.json" ]
}

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

set -eo pipefail
exec 2>&1
echo "[sbatch] host=\$(hostname) pid=\$\$ date=\$(date -Iseconds)"
echo "[sbatch] SLURM_JOB_ID=\${SLURM_JOB_ID:-n/a}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -3 || true

source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniforge 2>/dev/null || true
eval "\$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
conda activate $CONDA_ENV
echo "[sbatch] CONDA_PREFIX=\${CONDA_PREFIX:-unset}"
python --version
python -c "import torch; print('[sbatch] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd "$PROJECT_ROOT"
echo "[sbatch] CMD=$CMD"

# Force unbuffered python so logs stream live
CMD_UNBUFFERED=\$(echo "$CMD" | sed -E 's|^(\s*)python |\1python -u |')
echo "[sbatch] CMD_UNBUFFERED=\$CMD_UNBUFFERED"

eval "\$CMD_UNBUFFERED"
RC=\$?
echo "[sbatch] exit_code=\$RC"
exit \$RC
EOF
    echo "Submitted: $JOB_NAME"
}

echo "=== Language vs Vision: ViT on CIFAR-10 ==="
for OPT in "${OPTIMIZERS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TAG="vit_${OPT}_s${SEED}"
        OUTDIR="$RESULTS_ROOT/vision/lang_vs_vision/$TAG"
        if has_result "$OUTDIR"; then
            echo "  [SKIP] $TAG (already done)"
            continue
        fi
        submit "vit_${OPT}_s${SEED}" "$SLURM_TIME" \
            python experiments/vision/train_vit_cifar.py \
                --output_dir "$RESULTS_ROOT/vision/lang_vs_vision" \
                --optimizer "$OPT" --seed "$SEED" \
                --max_iters 5000 --batch_size 64 \
                --spectral_log_every 100 --spectral_full_svd True
    done
done

echo ""
echo "=== Submission complete ==="
echo "Total jobs submitted: $TOTAL"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Status:  bash experiments/nanogpt/check_cluster_status.sh"
