#!/bin/bash
# =============================================================
# Muon Implicit Bias — Full Experiment Suite
# Usage:
#   bash all_jobs.sh              # Run locally (sequential)
#   bash all_jobs.sh --parallel   # Run locally (parallel)
#   bash all_jobs.sh --slurm      # Submit as SLURM array jobs
# =============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # project root

RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR/matrix_sensing" "$RESULTS_DIR/nanogpt" "$RESULTS_DIR/plots"

MODE="${1:-sequential}"

echo "=========================================="
echo "  Muon Experiment Suite"
echo "  Mode: $MODE"
echo "  Results: $RESULTS_DIR"
echo "=========================================="

# ------------------------------------------------------------------
# Helper: run a matrix-sensing experiment
# ------------------------------------------------------------------
run_ms() {
    python experiments/matrix_sensing/$1 --output_dir "$RESULTS_DIR/matrix_sensing" "${@:2}"
}

# ------------------------------------------------------------------
# Helper: run a NanoGPT experiment
# ------------------------------------------------------------------
run_ng() {
    python experiments/nanogpt/$1 --output_dir "$RESULTS_DIR/nanogpt" "${@:2}"
}

# === STAGE 1: MATRIX SENSING (CPU only, ~2 hours) =================
echo ""
echo "[Stage 1] Matrix Sensing Experiments"
echo "--------------------------------------"

if [ "$MODE" = "parallel" ] || [ "$MODE" = "--parallel" ]; then

    echo "  Group 1/4: core four-way, LR sweep, kappa scaling"
    run_ms 01_e1_four_way.py       --n_seeds 20 &
    run_ms 02_e1_lr_sweep.py       --n_seeds 20 &
    run_ms 03_kappa_scaling.py     --n_seeds 10 &
    wait

    echo "  Group 2/4: ablation, block, factorial"
    run_ms 04_1500_config_ablation.py --n_seeds 10 &
    run_ms 05_block_acquisition.py    --n_seeds 10 &
    run_ms 06_block_k_sweep.py        --n_seeds 10 &
    run_ms 07_factorial_2x2.py        --n_seeds 10 &
    wait

    echo "  Group 3/4: alignment, attractor, weight-decay, exact"
    run_ms 08_alignment_tracking.py --n_seeds 5  &
    run_ms 09_spectral_attractor.py --n_seeds 10 &
    run_ms 10_weight_decay_atsr.py  --n_seeds 10 &
    run_ms 11_exact_trA.py          --n_seeds 5  &
    run_ms 12_entropy_floor.py      --n_seeds 5  &
    wait

    echo "  Group 4/4: alternative opts, dimension, noise"
    run_ms 13_alternative_optimizers.py --n_seeds 10 &
    run_ms 14_dimension_scaling.py      --n_seeds 10 &
    run_ms 15_noise_robustness.py       --n_seeds 10 &
    wait

elif [ "$MODE" = "--slurm" ]; then

    echo "  Submitting SLURM jobs for matrix sensing..."
    for script in 01_e1_four_way.py 02_e1_lr_sweep.py 03_kappa_scaling.py \
                  04_1500_config_ablation.py 05_block_acquisition.py 06_block_k_sweep.py \
                  07_factorial_2x2.py 08_alignment_tracking.py 09_spectral_attractor.py \
                  10_weight_decay_atsr.py 11_exact_trA.py 12_entropy_floor.py \
                  13_alternative_optimizers.py 14_dimension_scaling.py 15_noise_robustness.py; do
        sbatch --cpus-per-task=4 --mem=8G --time=02:00:00 \
            --job-name="ms_${script%.py}" \
            --wrap="python experiments/matrix_sensing/$script --output_dir $RESULTS_DIR/matrix_sensing --n_seeds 20"
        echo "    Submitted $script"
    done

else

    # Sequential (default)
    for script in 01_e1_four_way.py 02_e1_lr_sweep.py 03_kappa_scaling.py \
                  04_1500_config_ablation.py 05_block_acquisition.py 06_block_k_sweep.py \
                  07_factorial_2x2.py 08_alignment_tracking.py 09_spectral_attractor.py \
                  10_weight_decay_atsr.py 11_exact_trA.py 12_entropy_floor.py \
                  13_alternative_optimizers.py 14_dimension_scaling.py 15_noise_robustness.py; do
        echo "  Running $script..."
        run_ms "$script" --n_seeds 10
    done
fi

echo "[Stage 1] Complete."

# === STAGE 2: NANOGPT (GPU required, ~24-48 hours) ================
echo ""
echo "[Stage 2] NanoGPT Experiments"
echo "--------------------------------------"

# Setup NanoGPT if needed
if [ ! -d "nanoGPT" ]; then
    echo "  Setting up NanoGPT..."
    bash experiments/nanogpt/setup.sh
fi

if [ "$MODE" = "--slurm" ]; then

    echo "  Submitting SLURM jobs..."

    # --- 01: Muon vs AdamW baseline ---
    for opt in muon adamw; do
        for seed in 1 2 3 4 5; do
            sbatch --gres=gpu:1 --time=01:00:00 \
                --job-name="ng01_${opt}_s${seed}" \
                --wrap="python experiments/nanogpt/01_muon_vs_adamw.py --optimizer $opt --seed $seed --output_dir $RESULTS_DIR/nanogpt"
        done
    done

    # --- 02: Batch-size sweep ---
    for bs in 8 16 32 64 128 256 512; do
        for opt in muon adamw; do
            for seed in 1 2 3; do
                sbatch --gres=gpu:1 --time=01:00:00 \
                    --job-name="ng02_bs${bs}_${opt}_s${seed}" \
                    --wrap="python experiments/nanogpt/02_batch_size_sweep.py --batch_size $bs --optimizer $opt --seed $seed --output_dir $RESULTS_DIR/nanogpt"
            done
        done
    done

    # --- 03: LR sweep ---
    for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1; do
        for bs in 16 32 64 128 256; do
            for opt in muon adamw; do
                sbatch --gres=gpu:1 --time=00:30:00 \
                    --job-name="ng03_lr${lr}_bs${bs}_${opt}" \
                    --wrap="python experiments/nanogpt/03_lr_sweep.py --lr $lr --batch_size $bs --optimizer $opt --seed 42 --output_dir $RESULTS_DIR/nanogpt"
            done
        done
    done

    # --- 04: Spectral tracking ---
    for opt in muon adamw; do
        sbatch --gres=gpu:1 --time=02:00:00 \
            --job-name="ng04_spec_${opt}" \
            --wrap="python experiments/nanogpt/04_spectral_tracking.py --optimizer $opt --output_dir $RESULTS_DIR/nanogpt"
    done

    # --- 05: Effective rank ---
    for opt in muon adamw; do
        sbatch --gres=gpu:1 --time=01:00:00 \
            --job-name="ng05_rank_${opt}" \
            --wrap="python experiments/nanogpt/05_effective_rank.py --optimizer $opt --output_dir $RESULTS_DIR/nanogpt"
    done

    # --- 06: Weight-decay ablation ---
    for wd in 0 0.01 0.1 0.3; do
        sbatch --gres=gpu:1 --time=01:00:00 \
            --job-name="ng06_wd${wd}" \
            --wrap="python experiments/nanogpt/06_weight_decay_ablation.py --weight_decay $wd --optimizer muon --output_dir $RESULTS_DIR/nanogpt"
    done

    # --- 07: Gradient noise ---
    for opt in muon adamw; do
        sbatch --gres=gpu:1 --time=01:00:00 \
            --job-name="ng07_noise_${opt}" \
            --wrap="python experiments/nanogpt/07_gradient_noise.py --optimizer $opt --output_dir $RESULTS_DIR/nanogpt"
    done

else

    # Local sequential / parallel mode
    echo "  Running NanoGPT experiments locally (this may take a while)..."

    for opt in muon adamw; do
        for seed in 1 2 3; do
            echo "    01_muon_vs_adamw: opt=$opt seed=$seed"
            run_ng 01_muon_vs_adamw.py --optimizer "$opt" --seed "$seed"
        done
    done

    for bs in 16 64 256; do
        for opt in muon adamw; do
            echo "    02_batch_size_sweep: bs=$bs opt=$opt"
            run_ng 02_batch_size_sweep.py --batch_size "$bs" --optimizer "$opt" --seed 42
        done
    done

    for lr in 1e-3 1e-2 1e-1; do
        for bs in 32 128; do
            for opt in muon adamw; do
                echo "    03_lr_sweep: lr=$lr bs=$bs opt=$opt"
                run_ng 03_lr_sweep.py --lr "$lr" --batch_size "$bs" --optimizer "$opt" --seed 42
            done
        done
    done

    for opt in muon adamw; do
        echo "    04_spectral_tracking: opt=$opt"
        run_ng 04_spectral_tracking.py --optimizer "$opt"
    done

    for opt in muon adamw; do
        echo "    05_effective_rank: opt=$opt"
        run_ng 05_effective_rank.py --optimizer "$opt"
    done

    for wd in 0 0.1; do
        echo "    06_weight_decay_ablation: wd=$wd"
        run_ng 06_weight_decay_ablation.py --weight_decay "$wd" --optimizer muon
    done

    for opt in muon adamw; do
        echo "    07_gradient_noise: opt=$opt"
        run_ng 07_gradient_noise.py --optimizer "$opt"
    done
fi

echo "[Stage 2] Complete."

# === STAGE 3: GENERATE ALL PLOTS ==================================
echo ""
echo "[Stage 3] Generating Plots"
echo "--------------------------------------"
python experiments/plots/plot_all.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir  "$RESULTS_DIR"
echo "[Stage 3] Complete."

echo ""
echo "=========================================="
echo "  All experiments complete!"
echo "  Results: $RESULTS_DIR/"
echo "  Plots:   $RESULTS_DIR/plots/"
echo "=========================================="
