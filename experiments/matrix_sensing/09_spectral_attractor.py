#!/usr/bin/env python3
"""Spectral attractor analysis.

Track H/H_max at convergence across conditions.
Vary rank (2, 5, 10, full) and kappa (3, 10).
Muon only. 10 seeds. Test if attractor value is rank-dependent.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats

from src.optimizers import muon_step
from src.metrics import spectral_entropy, max_entropy
from src.matrix_sensing import generate_problem, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def main():
    parser = get_parser("Spectral attractor: H/H_max at convergence vs rank and kappa")
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--p', type=int, default=300)
    parser.add_argument('--n_steps', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    args.n_seeds = 10

    ranks = [2, 5, 10, min(args.m, args.n)]  # last is "full rank"
    kappas = [3.0, 10.0]
    H_max = max_entropy(args.m, args.n)

    condition_results = {}

    for rank in ranks:
        for kappa in kappas:
            key = f"rank={rank}_kappa={kappa}"
            print(f"\n=== {key} ===")

            H_converged = []
            for s_idx in range(args.n_seeds):
                seed = args.seed + s_idx
                set_seed(seed)
                W_star, A, b = generate_problem(args.m, args.n, args.p, rank,
                                                kappa=kappa, seed=seed)

                set_seed(seed + 10000)
                W = np.random.randn(args.m, args.n) * 0.1
                mom_buf = None

                for t in range(args.n_steps):
                    _, G = compute_loss_and_gradient(W, A, b)
                    W, mom_buf = muon_step(W, G, args.lr)

                H = spectral_entropy(W)
                H_converged.append(float(H))

            arr = np.array(H_converged)
            condition_results[key] = {
                'rank': rank,
                'kappa': kappa,
                'H_values': H_converged,
                'H_mean': float(np.mean(arr)),
                'H_std': float(np.std(arr, ddof=1)),
                'H_over_Hmax_mean': float(np.mean(arr / H_max)),
                'H_over_Hmax_std': float(np.std(arr / H_max, ddof=1)),
            }
            print(f"  H/H_max = {np.mean(arr)/H_max:.4f} +/- {np.std(arr, ddof=1)/H_max:.4f}")

    # Test rank-dependence: one-way ANOVA across ranks (pooling kappas)
    groups_by_rank = {}
    for rank in ranks:
        vals = []
        for kappa in kappas:
            key = f"rank={rank}_kappa={kappa}"
            vals.extend(condition_results[key]['H_values'])
        groups_by_rank[rank] = np.array(vals) / H_max

    anova_groups = [groups_by_rank[r] for r in ranks]
    f_stat, p_anova = stats.f_oneway(*anova_groups)

    # Test kappa-dependence at fixed rank
    kappa_tests = {}
    for rank in ranks:
        k3 = np.array(condition_results[f"rank={rank}_kappa=3.0"]['H_values']) / H_max
        k10 = np.array(condition_results[f"rank={rank}_kappa=10.0"]['H_values']) / H_max
        t_stat, p_val = stats.ttest_ind(k3, k10)
        kappa_tests[f"rank={rank}"] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'kappa3_mean': float(np.mean(k3)),
            'kappa10_mean': float(np.mean(k10)),
        }

    print(f"\n=== Rank-dependence ANOVA ===")
    print(f"F={f_stat:.4f}, p={p_anova:.4e}")
    for rank in ranks:
        print(f"  rank={rank}: H/H_max = {np.mean(groups_by_rank[rank]):.4f}")

    results = {
        'experiment': 'spectral_attractor',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p,
            'ranks': ranks, 'kappas': kappas,
            'n_steps': args.n_steps, 'lr': args.lr,
            'n_seeds': args.n_seeds, 'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'conditions': condition_results,
        'rank_dependence_anova': {
            'F_statistic': float(f_stat),
            'p_value': float(p_anova),
            'per_rank_mean_H_over_Hmax': {
                str(r): float(np.mean(groups_by_rank[r])) for r in ranks
            },
        },
        'kappa_dependence_tests': kappa_tests,
    }

    save_results(results, args.output_dir, '09_spectral_attractor.json')


if __name__ == '__main__':
    main()
