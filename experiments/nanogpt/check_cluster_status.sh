#!/usr/bin/env bash
# =============================================================================
# check_cluster_status.sh
#
# Run ON THE MIT ENGAGING CLUSTER to monitor NanoGPT experiment status.
# Checks SLURM job queue and counts completed runs per experiment.
# =============================================================================
set -euo pipefail

RESULTS_DIR="$HOME/projects/muon/results/nanogpt"
REMOTE_HOST="pierb@orcd-login001.mit.edu"
LOCAL_DEST="~/Desktop/Experiments/PoggioAI-results/project_003_muon/experiments/results/nanogpt/"

# Experiment directories to check (01-12, skipping 09 and 11 which don't exist)
EXPERIMENTS=(
    "01_muon_vs_adamw"
    "02_batch_size_sweep"
    "03_lr_sweep"
    "04_spectral_tracking"
    "05_feature_acquisition"
    "06_weight_decay_ablation"
    "07_momentum_ablation"
    "08_model_scale"
    "10_head_ablation"
    "12_regression_vs_cls"
)

# ── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# =============================================================================
# 1. SLURM Job Status
# =============================================================================
echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  NanoGPT Cluster Status Check${RESET}"
echo -e "${BOLD}  $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

echo -e "${CYAN}--- SLURM Queue ---${RESET}"
RUNNING=$(squeue -u "$USER" -h -t RUNNING 2>/dev/null | wc -l | tr -d ' ')
PENDING=$(squeue -u "$USER" -h -t PENDING 2>/dev/null | wc -l | tr -d ' ')
TOTAL_QUEUED=$((RUNNING + PENDING))

echo -e "  Running:  ${GREEN}${RUNNING}${RESET}"
echo -e "  Pending:  ${YELLOW}${PENDING}${RESET}"
echo -e "  Total:    ${BOLD}${TOTAL_QUEUED}${RESET}"
echo ""

# Show job details if any are queued
if [ "$TOTAL_QUEUED" -gt 0 ]; then
    echo -e "${CYAN}--- Active Jobs (first 20) ---${RESET}"
    squeue -u "$USER" -o "%.8i %.30j %.8T %.10M %.6D" 2>/dev/null | head -21
    echo ""
fi

# =============================================================================
# 2. Per-Experiment Completion Status
# =============================================================================
echo -e "${CYAN}--- Experiment Completion ---${RESET}"
echo ""
printf "  ${BOLD}%-30s  %8s  %8s  %8s  %s${RESET}\n" \
    "EXPERIMENT" "DONE" "TOTAL" "PERCENT" "STATUS"
printf "  %-30s  %8s  %8s  %8s  %s\n" \
    "------------------------------" "--------" "--------" "--------" "------"

TOTAL_DONE=0
TOTAL_RUNS=0
ALL_COMPLETE=true

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="${RESULTS_DIR}/${EXP}"

    if [ ! -d "$EXP_DIR" ]; then
        printf "  %-30s  %8s  %8s  %8s  ${RED}%s${RESET}\n" \
            "$EXP" "-" "-" "-" "NO DIR"
        ALL_COMPLETE=false
        continue
    fi

    # Count total subdirectories (each is one run)
    N_TOTAL=$(find "$EXP_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

    # Count runs that have a summary.json (completed)
    N_DONE=$(find "$EXP_DIR" -mindepth 2 -maxdepth 2 -name "summary.json" | wc -l | tr -d ' ')

    TOTAL_DONE=$((TOTAL_DONE + N_DONE))
    TOTAL_RUNS=$((TOTAL_RUNS + N_TOTAL))

    if [ "$N_TOTAL" -eq 0 ]; then
        PCT="0"
        STATUS_COLOR="$RED"
        STATUS_TEXT="EMPTY"
        ALL_COMPLETE=false
    elif [ "$N_DONE" -eq "$N_TOTAL" ]; then
        PCT="100"
        STATUS_COLOR="$GREEN"
        STATUS_TEXT="DONE"
    else
        PCT=$(( (N_DONE * 100) / N_TOTAL ))
        STATUS_COLOR="$YELLOW"
        STATUS_TEXT="RUNNING"
        ALL_COMPLETE=false
    fi

    printf "  %-30s  %8d  %8d  %7d%%  ${STATUS_COLOR}%s${RESET}\n" \
        "$EXP" "$N_DONE" "$N_TOTAL" "$PCT" "$STATUS_TEXT"
done

echo ""
printf "  ${BOLD}%-30s  %8d  %8d  %7d%%${RESET}\n" \
    "TOTAL" "$TOTAL_DONE" "$TOTAL_RUNS" \
    "$(( TOTAL_RUNS > 0 ? (TOTAL_DONE * 100) / TOTAL_RUNS : 0 ))"
echo ""

# =============================================================================
# 3. List Incomplete Runs (if any)
# =============================================================================
if ! $ALL_COMPLETE; then
    echo -e "${CYAN}--- Incomplete Runs ---${RESET}"
    INCOMPLETE_COUNT=0
    for EXP in "${EXPERIMENTS[@]}"; do
        EXP_DIR="${RESULTS_DIR}/${EXP}"
        [ ! -d "$EXP_DIR" ] && continue

        for RUN_DIR in "$EXP_DIR"/*/; do
            [ ! -d "$RUN_DIR" ] && continue
            if [ ! -f "${RUN_DIR}summary.json" ]; then
                echo "  $(basename "$EXP")/$(basename "$RUN_DIR")"
                INCOMPLETE_COUNT=$((INCOMPLETE_COUNT + 1))
            fi
        done
    done
    if [ "$INCOMPLETE_COUNT" -eq 0 ]; then
        echo "  (none found -- directories may not exist yet)"
    fi
    echo ""
fi

# =============================================================================
# 4. Rsync Command
# =============================================================================
if $ALL_COMPLETE && [ "$TOTAL_QUEUED" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All experiments complete and no jobs in queue!${RESET}"
    echo ""
    echo -e "${BOLD}Download results with:${RESET}"
    echo ""
    echo "  rsync -avz ${REMOTE_HOST}:~/projects/muon/results/nanogpt/ ${LOCAL_DEST}"
    echo ""
else
    echo -e "${YELLOW}Jobs still in progress. When all complete, download with:${RESET}"
    echo ""
    echo "  rsync -avz ${REMOTE_HOST}:~/projects/muon/results/nanogpt/ ${LOCAL_DEST}"
    echo ""
fi
