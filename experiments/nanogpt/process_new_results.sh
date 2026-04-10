#!/usr/bin/env bash
# =============================================================================
# process_new_results.sh
#
# Run LOCALLY after rsyncing results from the cluster.
# Counts completed runs, identifies new results since last check,
# and suggests which plot scripts to re-run.
# =============================================================================
set -euo pipefail

RESULTS_DIR="/Users/pier-pc4/Desktop/Experiments/PoggioAI-results/project_003_muon/experiments/results/nanogpt"
LAST_CHECK_FILE="${RESULTS_DIR}/.last_check"

# Experiment directories and their associated plot scripts
declare -A PLOT_MAP
PLOT_MAP=(
    ["01_muon_vs_adamw"]="plot_nanogpt_bcrit.py"
    ["02_batch_size_sweep"]="plot_nanogpt_bcrit.py"
    ["03_lr_sweep"]="plot_nanogpt_bcrit.py"
    ["04_spectral_tracking"]="plot_nanogpt_spectral.py"
    ["05_feature_acquisition"]="plot_nanogpt_spectral.py"
    ["06_weight_decay_ablation"]="plot_nanogpt_ablations.py"
    ["07_momentum_ablation"]="plot_nanogpt_ablations.py"
    ["08_model_scale"]="plot_nanogpt_ablations.py"
    ["10_head_ablation"]="plot_nanogpt_ablations.py"
    ["12_regression_vs_cls"]="(standalone analysis)"
)

# Ordered list for consistent output
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
# 0. Check results directory exists
# =============================================================================
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Results directory not found: ${RESULTS_DIR}${RESET}" >&2
    echo "Run rsync to download results first." >&2
    exit 1
fi

echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  NanoGPT Local Results Summary${RESET}"
echo -e "${BOLD}  $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

# =============================================================================
# 1. Determine last check time
# =============================================================================
if [ -f "$LAST_CHECK_FILE" ]; then
    LAST_CHECK=$(cat "$LAST_CHECK_FILE")
    LAST_CHECK_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "$LAST_CHECK" +%s 2>/dev/null || echo "0")
    echo -e "${CYAN}Last check:${RESET} $LAST_CHECK"
else
    LAST_CHECK="(never)"
    LAST_CHECK_EPOCH=0
    echo -e "${CYAN}Last check:${RESET} (first run)"
fi
echo ""

# =============================================================================
# 2. Per-Experiment Status Table
# =============================================================================
echo -e "${CYAN}--- Experiment Status ---${RESET}"
echo ""
printf "  ${BOLD}%-30s  %8s  %8s  %10s  %8s${RESET}\n" \
    "EXPERIMENT" "DONE" "TOTAL" "COMPLETE" "NEW"
printf "  %-30s  %8s  %8s  %10s  %8s\n" \
    "------------------------------" "--------" "--------" "----------" "--------"

TOTAL_DONE=0
TOTAL_RUNS=0
TOTAL_NEW=0

# Track which experiments have new data (for plot suggestions)
declare -A HAS_NEW_DATA

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="${RESULTS_DIR}/${EXP}"

    if [ ! -d "$EXP_DIR" ]; then
        printf "  %-30s  %8s  %8s  %10s  %8s\n" "$EXP" "-" "-" "NO DIR" "-"
        continue
    fi

    # Count total run subdirectories
    N_TOTAL=$(find "$EXP_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

    # Count completed runs (have summary.json)
    N_DONE=$(find "$EXP_DIR" -mindepth 2 -maxdepth 2 -name "summary.json" | wc -l | tr -d ' ')

    # Count new summary.json files since last check
    N_NEW=0
    if [ "$LAST_CHECK_EPOCH" -gt 0 ]; then
        N_NEW=$(find "$EXP_DIR" -mindepth 2 -maxdepth 2 -name "summary.json" \
            -newer "$LAST_CHECK_FILE" 2>/dev/null | wc -l | tr -d ' ')
    else
        # First run: everything is "new"
        N_NEW=$N_DONE
    fi

    TOTAL_DONE=$((TOTAL_DONE + N_DONE))
    TOTAL_RUNS=$((TOTAL_RUNS + N_TOTAL))
    TOTAL_NEW=$((TOTAL_NEW + N_NEW))

    if [ "$N_NEW" -gt 0 ]; then
        HAS_NEW_DATA["$EXP"]=1
    fi

    # Format completion percentage
    if [ "$N_TOTAL" -eq 0 ]; then
        PCT_STR="0%"
    else
        PCT=$(( (N_DONE * 100) / N_TOTAL ))
        PCT_STR="${PCT}%"
    fi

    # Color the status
    if [ "$N_TOTAL" -gt 0 ] && [ "$N_DONE" -eq "$N_TOTAL" ]; then
        STATUS_COLOR="$GREEN"
    elif [ "$N_DONE" -gt 0 ]; then
        STATUS_COLOR="$YELLOW"
    else
        STATUS_COLOR="$RED"
    fi

    # Color the new count
    if [ "$N_NEW" -gt 0 ]; then
        NEW_STR="${GREEN}+${N_NEW}${RESET}"
    else
        NEW_STR="-"
    fi

    printf "  %-30s  %8d  %8d  ${STATUS_COLOR}%10s${RESET}  %8b\n" \
        "$EXP" "$N_DONE" "$N_TOTAL" "$PCT_STR" "$NEW_STR"
done

echo ""
printf "  ${BOLD}%-30s  %8d  %8d  %10s  +%d${RESET}\n" \
    "TOTAL" "$TOTAL_DONE" "$TOTAL_RUNS" \
    "$(( TOTAL_RUNS > 0 ? (TOTAL_DONE * 100) / TOTAL_RUNS : 0 ))%" \
    "$TOTAL_NEW"
echo ""

# =============================================================================
# 3. Detailed Per-Experiment Run Status
# =============================================================================
echo -e "${CYAN}--- Per-Experiment Run Details ---${RESET}"
echo ""

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="${RESULTS_DIR}/${EXP}"
    [ ! -d "$EXP_DIR" ] && continue

    SUCCEEDED=()
    FAILED=()

    for RUN_DIR in "$EXP_DIR"/*/; do
        [ ! -d "$RUN_DIR" ] && continue
        RUN_NAME=$(basename "$RUN_DIR")

        if [ -f "${RUN_DIR}summary.json" ]; then
            # Check if this is a new result
            if [ "$LAST_CHECK_EPOCH" -gt 0 ] && [ "$LAST_CHECK_FILE" -ot "${RUN_DIR}summary.json" ]; then
                SUCCEEDED+=("${RUN_NAME} ${GREEN}(new)${RESET}")
            else
                SUCCEEDED+=("$RUN_NAME")
            fi
        else
            FAILED+=("$RUN_NAME")
        fi
    done

    N_SUCC=${#SUCCEEDED[@]}
    N_FAIL=${#FAILED[@]}

    echo -e "  ${BOLD}${EXP}${RESET}  (${GREEN}${N_SUCC} ok${RESET}, ${RED}${N_FAIL} missing${RESET})"

    if [ "$N_FAIL" -gt 0 ]; then
        for f in "${FAILED[@]}"; do
            echo -e "    ${RED}x${RESET} $f"
        done
    fi

    # Only show new results in detail to keep output manageable
    NEW_SHOWN=false
    for s in "${SUCCEEDED[@]}"; do
        if [[ "$s" == *"(new)"* ]]; then
            echo -e "    ${GREEN}+${RESET} $s"
            NEW_SHOWN=true
        fi
    done

    if ! $NEW_SHOWN && [ "$N_FAIL" -eq 0 ]; then
        echo -e "    ${GREEN}All ${N_SUCC} runs complete${RESET}"
    fi

    echo ""
done

# =============================================================================
# 4. New Results Since Last Check
# =============================================================================
if [ "$TOTAL_NEW" -gt 0 ]; then
    echo -e "${CYAN}--- New Since Last Check ---${RESET}"
    echo ""
    echo -e "  ${BOLD}${TOTAL_NEW} new summary.json files${RESET} found since ${LAST_CHECK}"
    echo ""

    if [ "$LAST_CHECK_EPOCH" -gt 0 ]; then
        echo "  New files:"
        find "$RESULTS_DIR" -name "summary.json" -newer "$LAST_CHECK_FILE" -print 2>/dev/null | \
            sort | while read -r f; do
            REL_PATH="${f#"$RESULTS_DIR"/}"
            echo "    $REL_PATH"
        done
        echo ""
    fi
fi

# =============================================================================
# 5. Plot Script Suggestions
# =============================================================================
echo -e "${CYAN}--- Suggested Plot Scripts to Re-run ---${RESET}"
echo ""

# Collect unique plot scripts that need updating
declare -A PLOTS_TO_RUN

for EXP in "${!HAS_NEW_DATA[@]}"; do
    PLOT="${PLOT_MAP[$EXP]}"
    if [ -n "$PLOT" ]; then
        # Append experiment name to the plot entry
        if [ -n "${PLOTS_TO_RUN[$PLOT]+x}" ]; then
            PLOTS_TO_RUN["$PLOT"]="${PLOTS_TO_RUN[$PLOT]}, ${EXP}"
        else
            PLOTS_TO_RUN["$PLOT"]="$EXP"
        fi
    fi
done

if [ ${#PLOTS_TO_RUN[@]} -eq 0 ]; then
    echo -e "  ${GREEN}No new data -- nothing to re-plot.${RESET}"
else
    for PLOT in $(echo "${!PLOTS_TO_RUN[@]}" | tr ' ' '\n' | sort -u); do
        echo -e "  ${BOLD}${PLOT}${RESET}"
        echo "    Triggered by: ${PLOTS_TO_RUN[$PLOT]}"
        echo ""
    done
fi
echo ""

# =============================================================================
# 6. Update Last Check Timestamp
# =============================================================================
date '+%Y-%m-%d %H:%M:%S' > "$LAST_CHECK_FILE"
echo -e "${GREEN}Updated last-check timestamp: $(cat "$LAST_CHECK_FILE")${RESET}"
echo ""
