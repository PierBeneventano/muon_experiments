#!/usr/bin/env python3
"""E1 comparison expanded with Adam, LARS, SignSGD.

Same setup as E1 (m=n=20, p=200, rank=2, 5000 steps).
Optimizers: Muon, GD, NM-GD, Random-Orth, Adam, LARS, SignSGD.
10 seeds each. Full pairwise comparison.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats
from itertools import combinations

from src.optimizers import (muon_step, gd_step, norm_matched_gd_step,
                            random_orthogonal_step, adam_step, lars_step,
                            sign_gd_step)
from src.metrics import spectral_entropy, max_entropy
from src.matrix_sensing import generate_problem, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def run_optimizer(optimizer_name, W_init, A, b, n_steps, lr):
    """Run optimizer, return final W."""
    W = W_init.copy()
    mom_buf = None
    m_buf = None
    v_buf = None

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
        elif optimizer_name == 'adam':
            W, m_buf, v_buf = adam_step(W, G, lr, t, m_buf, v_buf)
        elif optimizer_name == 'lars':
            W = lars_step(W, G, lr)
        elif optimizer_name == 'sign_gd':
            W = sign_gd_step(W, G, lr)

    return W


def main():
    parser = get_parser("Extended optimizer comparison (7 optimizers)")
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    args.n_seeds = 10

    optimizers = ['muon', 'gd', 'nm_gd', 'random_orth', 'adam', 'lars', 'sign_gd']
    labels = {
        'muon': 'Muon', 'gd': 'GD', 'nm_gd': 'NM-GD',
        'random_orth': 'Random-Orth', 'adam': 'Adam',
        'lars': 'LARS', 'sign_gd': 'SignSGD',
    }
    # Per-optimizer learning rates (some need different scales)
    opt_lrs = {
        'muon': args.lr, 'gd': args.lr, 'nm_gd': args.lr,
        'random_orth': args.lr, 'adam': 0.001,
        'lars': args.lr, 'sign_gd': 0.001,
    }
    H_max = max_entropy(args.m, args.n)

    H_values = {opt: [] for opt in optimizers}

    for s_idx in range(args.n_seeds):
        seed = args.seed + s_idx
        print(f"Seed {s_idx+1}/{args.n_seeds} (seed={seed})")

        set_seed(seed)
        W_star, A, b = generate_problem(args.m, args.n, args.p, args.rank, seed=seed)

        for opt in optimizers:
            set_seed(seed + 10000)
            W_init = np.random.randn(args.m, args.n) * 0.1
            W_final = run_optimizer(opt, W_init, A, b, args.n_steps, opt_lrs[opt])

            if np.any(np.isnan(W_final)) or np.any(np.isinf(W_final)):
                H = 0.0
                print(f"  {labels[opt]}: DIVERGED")
            else:
                H = spectral_entropy(W_final)
                print(f"  {labels[opt]}: H/H_max={H/H_max:.4f}")

            H_values[opt].append(float(H))

    # Pairwise tests (Bonferroni-corrected)
    n_comparisons = len(list(combinations(optimizers, 2)))
    pairwise = {}
    for opt_a, opt_b in combinations(optimizers, 2):
        h_a = np.array(H_values[opt_a])
        h_b = np.array(H_values[opt_b])
        # Use paired test when possible (same seeds)
        t_stat, p_val = stats.ttest_rel(h_a, h_b)
        p_corr = min(p_val * n_comparisons, 1.0)
        diff = h_a - h_b
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-15)
        key = f"{labels[opt_a]}_vs_{labels[opt_b]}"
        pairwise[key] = {
            't_stat': float(t_stat),
            'p_raw': float(p_val),
            'p_bonferroni': float(p_corr),
            'cohens_d': float(d),
            'mean_diff': float(np.mean(diff)),
        }

    # Summary
    summary = {}
    for opt in optimizers:
        arr = np.array(H_values[opt])
        summary[labels[opt]] = {
            'lr': opt_lrs[opt],
            'H_mean': float(np.mean(arr)),
            'H_std': float(np.std(arr, ddof=1)),
            'H_over_Hmax_mean': float(np.mean(arr / H_max)),
            'H_over_Hmax_std': float(np.std(arr / H_max, ddof=1)),
        }

    # Rank optimizers by mean H
    ranked = sorted(summary.items(), key=lambda x: x[1]['H_mean'], reverse=True)
    ranking = [{'rank': i+1, 'optimizer': name, 'H_over_Hmax': s['H_over_Hmax_mean']}
               for i, (name, s) in enumerate(ranked)]

    print(f"\n=== Ranking by H/H_max ===")
    for r in ranking:
        print(f"  #{r['rank']}: {r['optimizer']} ({r['H_over_Hmax']:.4f})")

    results = {
        'experiment': 'alternative_optimizers',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p, 'rank': args.rank,
            'n_steps': args.n_steps, 'n_seeds': args.n_seeds,
            'base_seed': args.seed, 'opt_lrs': {labels[k]: v for k, v in opt_lrs.items()},
        },
        'H_max': float(H_max),
        'H_values': {labels[opt]: H_values[opt] for opt in optimizers},
        'summary': summary,
        'ranking': ranking,
        'pairwise_tests': pairwise,
        'bonferroni_n_comparisons': n_comparisons,
    }

    save_results(results, args.output_dir, '13_alternative_optimizers.json')


if __name__ == '__main__':
    main()
