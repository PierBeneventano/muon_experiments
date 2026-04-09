# Deploying to MIT Engaging Cluster

## Quick Version (copy-paste these commands)

### Step 1: On your laptop — push to GitHub

```bash
cd ~/Desktop/Experiments/PoggioAI-results/project_003_muon

# Init git repo if not already
git init
git add experiments/
git commit -m "Muon experiment suite: 15 matrix sensing + 12 NanoGPT + plots"
git remote add origin git@github.com:YOUR_USERNAME/muon-experiments.git
git push -u origin main
```

### Step 2: SSH to Engaging and clone

```bash
ssh YOUR_USER@orcd-login001.mit.edu

# Set up workspace
mkdir -p ~/projects/muon
cd ~/projects/muon
git clone git@github.com:YOUR_USERNAME/muon-experiments.git .
```

### Step 3: Set up conda environment

```bash
module load miniforge
conda create -n muon python=3.10 -y
conda activate muon
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib pyyaml tiktoken datasets transformers
```

### Step 4: Set up NanoGPT

```bash
cd ~/projects/muon
bash experiments/nanogpt/setup.sh
```

### Step 5: Run matrix sensing (CPU, interactive — optional, already done locally)

```bash
# These already ran on your laptop. Skip unless you want cluster-reproducible results.
srun -p mit_normal -t 02:00:00 --mem=16G -c 8 bash -c "
  cd ~/projects/muon
  conda activate muon
  bash experiments/all_jobs.sh --parallel
"
```

### Step 6: Submit NanoGPT GPU jobs

```bash
cd ~/projects/muon

# Set cluster-specific vars
export SLURM_PARTITION=mit_normal_gpu
export SLURM_ACCOUNT=""  # leave empty unless your PI requires it

# Dry run first to see all commands
bash experiments/nanogpt/all_jobs.sh --dry-run

# Submit for real (~281 jobs)
bash experiments/nanogpt/all_jobs.sh
```

### Step 7: Monitor

```bash
squeue -u $USER              # see running/queued jobs
squeue -u $USER | wc -l      # count remaining
sacct -j JOBID --format=Elapsed,State  # check specific job
ls results/slurm_logs/       # check logs
```

### Step 8: After all jobs complete — generate plots

```bash
cd ~/projects/muon
conda activate muon
python experiments/plots/plot_all.py \
  --results_dir experiments/results \
  --output_dir experiments/results/plots
```

### Step 9: Pull results back to laptop

```bash
# On your laptop:
rsync -avz YOUR_USER@orcd-login001.mit.edu:~/projects/muon/experiments/results/ \
  ~/Desktop/Experiments/PoggioAI-results/project_003_muon/experiments/results/
```

---

## Estimated Runtime

| Experiment | Jobs | GPU-hours (each ~2-10 min) | Wall time (parallel) |
|-----------|------|---------------------------|---------------------|
| 01 Muon vs AdamW | 6 | 0.2h | 10 min |
| 02 Batch size sweep | 42 | 1.4h | 10 min |
| 03 LR sweep | 98 | 3.3h | 10 min |
| 04 Spectral tracking | 6 | 0.2h | 10 min |
| 05 Feature acquisition | 6 | 0.2h | 10 min |
| 06 Weight decay | 24 | 0.8h | 10 min |
| 07 Momentum | 15 | 0.5h | 10 min |
| 08 Model scale | 24 | 2.4h | 30 min |
| 09 Depth | 24 | 2.4h | 30 min |
| 10 Heads | 24 | 0.8h | 10 min |
| 12 Reg vs cls | 12 | 0.4h | 10 min |
| **Total** | **281** | **~13 GPU-hours** | **~30 min** (if all parallel) |

Most jobs take 2-5 minutes on a single GPU. The bottleneck is queue wait time, not compute.

---

## Troubleshooting

**"module: command not found"**: Run `source /etc/profile.d/modules.sh` first.

**CUDA version mismatch**: Check `nvidia-smi` on compute nodes. Install matching PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118  # for CUDA 11.8
```

**"No module named tiktoken"**: `pip install tiktoken` in your conda env.

**Jobs stuck in PENDING**: Check `squeue -u $USER -l` for reasons. Common: partition full, resource limits exceeded. Try `--partition=mit_preemptable` for lower-priority but faster scheduling.

**Results not appearing**: Check `results/slurm_logs/` for error files. Common issues: wrong working directory, missing conda activation.
