# Muon Implicit Bias -- Experiment Suite

Reproducibility code for the paper on the implicit spectral bias of the Muon optimizer.

## Structure

```
experiments/
  matrix_sensing/   15 CPU experiments (01-15) on synthetic matrix-sensing
  nanogpt/          7 GPU experiments on character-level NanoGPT
  plots/            Plotting scripts (one per figure)
  results/          Auto-generated outputs and figures
  all_jobs.sh       Master runner (sequential, parallel, or SLURM)
  requirements.txt  Python dependencies
```

## Quick start

```bash
pip install -r experiments/requirements.txt

# Run everything sequentially (CPU experiments only take ~2 h)
bash experiments/all_jobs.sh

# Or parallelise CPU work
bash experiments/all_jobs.sh --parallel

# Submit GPU jobs on a SLURM cluster
bash experiments/all_jobs.sh --slurm
```

## Generating plots only

If you already have results in `experiments/results/`:

```bash
python experiments/plots/plot_all.py \
    --results_dir experiments/results \
    --output_dir  experiments/results
```

Figures are saved as PDF and PNG in `experiments/results/plots/`.
