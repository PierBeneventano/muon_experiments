#!/usr/bin/env python3
"""E1 four-way optimizer comparison on matrix sensing.

m=n=20, p=200, rank=2, 5000 steps.
Muon / GD / NM-GD / Random-Orth.
20 seeds. Measure spectral entropy H at convergence.
Paired t-tests with Bonferroni correction.
Save per-optimizer H values, pairwise p-values, Cohen's d.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats
from itertools import combinations

from src.optimizers import muon_step, gd_step, norm_matched_gd_step, random_orthogonal_step
from src.metrics import spectral_entropy, max_entropy
from src.matrix_sensing import generate_problem, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def run_optimizer(optimizer_name, W_init, A, b, n_steps, lr):
    """Run a single optimizer and return final W."""
    W = W_init.copy()
    mom_buf = None

    for t in range(n_steps):
        _, G = compute_loss_and_gradient(W, A, b)

        if optimizer_name == 'muon':
            W, mom_buf = muon_step(W, G, lr)
        elif optimizer_name == 'gd':
            W = gd_step(W, G, lr)
        elif optimizer_name == 'nm_gd':
            W = norm_matched_gd_step(W, G, lr)
        elif optimizer_name == 'random_orth':
            W = random_orthogonal_step(W, G, lr)

    return W


def main():
    parser = get_parser("E1 four-way optimizer comparison on matrix sensing")
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    args.n_seeds = 20  # Fixed for this experiment

    optimizers = ['muon', 'gd', 'nm_gd', 'random_orth']
    optimizer_labels = {'muon': 'Muon', 'gd': 'GD', 'nm_gd': 'NM-GD', 'random_orth': 'Random-Orth'}
    H_max = max_entropy(args.m, args.n)

    # Collect H values per optimizer per seed
    H_values = {opt: [] for opt in optimizers}

    for seed_idx in range(args.n_seeds):
        seed = args.seed + seed_idx
        print(f"Seed {seed_idx+1}/{args.n_seeds} (seed={seed})")

        # Generate problem (same for all optimizers within a seed)
        set_seed(seed)
        W_star, A, b = generate_problem(args.m, args.n, args.p, args.rank, seed=seed)

        for opt in optimizers:
            set_seed(seed + 10000)  # Same init across optimizers
            W_init = np.random.randn(args.m, args.n) * 0.1
            W_final = run_optimizer(opt, W_init, A, b, args.n_steps, args.lr)
            H = spectral_entropy(W_final)
            H_values[opt].append(float(H))
            print(f"  {optimizer_labels[opt]}: H={H:.4f} (H/H_max={H/H_max:.4f})")

    # Compute pairwise paired t-tests with Bonferroni correction
    n_comparisons = len(list(combinations(optimizers, 2)))
    pairwise_results = {}
    for opt_a, opt_b in combinations(optimizers, 2):
        h_a = np.array(H_values[opt_a])
        h_b = np.array(H_values[opt_b])
        t_stat, p_val = stats.ttest_rel(h_a, h_b)
        # Bonferroni correction
        p_corrected = min(p_val * n_comparisons, 1.0)
        # Cohen's d for paired samples
        diff = h_a - h_b
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-15)
        key = f"{optimizer_labels[opt_a]}_vs_{optimizer_labels[opt_b]}"
        pairwise_results[key] = {
            't_statistic': float(t_stat),
            'p_value_raw': float(p_val),
            'p_value_bonferroni': float(p_corrected),
            'cohens_d': float(d),
            'mean_diff': float(np.mean(diff)),
        }

    # Summary statistics
    summary = {}
    for opt in optimizers:
        h_arr = np.array(H_values[opt])
        summary[optimizer_labels[opt]] = {
            'H_mean': float(np.mean(h_arr)),
            'H_std': float(np.std(h_arr, ddof=1)),
            'H_over_Hmax_mean': float(np.mean(h_arr / H_max)),
            'H_over_Hmax_std': float(np.std(h_arr / H_max, ddof=1)),
        }

    results = {
        'experiment': 'E1_four_way_comparison',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p, 'rank': args.rank,
            'n_steps': args.n_steps, 'lr': args.lr, 'n_seeds': args.n_seeds,
            'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'H_values_per_optimizer': {optimizer_labels[opt]: H_values[opt] for opt in optimizers},
        'summary': summary,
        'pairwise_tests': pairwise_results,
        'bonferroni_n_comparisons': n_comparisons,
    }

    save_results(results, args.output_dir, '01_e1_four_way.json')
    print("\n=== Summary ===")
    for opt in optimizers:
        s = summary[optimizer_labels[opt]]
        print(f"  {optimizer_labels[opt]}: H/H_max = {s['H_over_Hmax_mean']:.4f} +/- {s['H_over_Hmax_std']:.4f}")


if __name__ == '__main__':
    main()
