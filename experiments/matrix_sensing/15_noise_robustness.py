#!/usr/bin/env python3
"""Noise robustness experiment.

Add isotropic noise sigma^2 in {0, 0.01, 0.1, 0.5, 1.0} to gradient.
Test if Muon's H advantage is robust to noise.
m=n=20, p=200, rank=2, 5000 steps. 10 seeds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats

from src.optimizers import muon_step, gd_step, polar
from src.metrics import spectral_entropy, max_entropy
from src.matrix_sensing import generate_problem, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def run_with_noise(optimizer_name, W_init, A, b, n_steps, lr, noise_std):
    """Run optimizer with additive Gaussian noise on the gradient."""
    W = W_init.copy()
    mom_buf = None

    for t in range(n_steps):
        _, G = compute_loss_and_gradient(W, A, b)

        # Add isotropic Gaussian noise to gradient
        if noise_std > 0:
            G = G + noise_std * np.random.randn(*G.shape)

        if optimizer_name == 'muon':
            W, mom_buf = muon_step(W, G, lr)
        elif optimizer_name == 'gd':
            W = gd_step(W, G, lr)

    return W


def main():
    parser = get_parser("Noise robustness: Muon vs GD under gradient noise")
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    args.n_seeds = 10

    noise_levels = [0.0, 0.01, 0.1, 0.5, 1.0]
    H_max = max_entropy(args.m, args.n)

    per_noise_results = {}

    for noise_std in noise_levels:
        print(f"\n=== noise_std={noise_std} ===")
        H_muon = []
        H_gd = []
        losses_muon = []
        losses_gd = []

        for s_idx in range(args.n_seeds):
            seed = args.seed + s_idx
            set_seed(seed)
            W_star, A, b = generate_problem(args.m, args.n, args.p, args.rank, seed=seed)

            for opt in ['muon', 'gd']:
                set_seed(seed + 10000)
                W_init = np.random.randn(args.m, args.n) * 0.1

                # Use a fixed noise seed per (seed, opt) pair for reproducibility
                np.random.seed(seed + 30000 + (0 if opt == 'muon' else 50000))
                W_final = run_with_noise(opt, W_init, A, b, args.n_steps, args.lr, noise_std)

                if np.any(np.isnan(W_final)) or np.any(np.isinf(W_final)):
                    H = 0.0
                    loss = float('inf')
                else:
                    H = spectral_entropy(W_final)
                    from src.matrix_sensing import compute_loss
                    loss = compute_loss(W_final, A, b)

                if opt == 'muon':
                    H_muon.append(float(H))
                    losses_muon.append(float(loss))
                else:
                    H_gd.append(float(H))
                    losses_gd.append(float(loss))

        muon_arr = np.array(H_muon)
        gd_arr = np.array(H_gd)
        adv = muon_arr - gd_arr

        # Paired t-test on H difference
        t_stat, p_val = stats.ttest_rel(muon_arr, gd_arr)

        per_noise_results[str(noise_std)] = {
            'noise_std': noise_std,
            'H_muon': H_muon,
            'H_gd': H_gd,
            'loss_muon_mean': float(np.mean(losses_muon)),
            'loss_gd_mean': float(np.mean(losses_gd)),
            'H_advantage_mean': float(np.mean(adv)),
            'H_advantage_std': float(np.std(adv, ddof=1)),
            'muon_H_over_Hmax': float(np.mean(muon_arr / H_max)),
            'gd_H_over_Hmax': float(np.mean(gd_arr / H_max)),
            'ttest_t': float(t_stat),
            'ttest_p': float(p_val),
            'advantage_significant': bool(p_val < 0.05),
        }

        print(f"  Muon H/H_max: {np.mean(muon_arr)/H_max:.4f} +/- {np.std(muon_arr, ddof=1)/H_max:.4f}")
        print(f"  GD   H/H_max: {np.mean(gd_arr)/H_max:.4f} +/- {np.std(gd_arr, ddof=1)/H_max:.4f}")
        print(f"  Advantage:    {np.mean(adv):.4f} (p={p_val:.4e})")

    # Does advantage degrade with noise?
    noise_arr = np.array(noise_levels)
    adv_means = np.array([per_noise_results[str(ns)]['H_advantage_mean'] for ns in noise_levels])
    rho, p_corr = stats.spearmanr(noise_arr, adv_means)

    # Fraction of noise levels where advantage is significant
    n_significant = sum(1 for ns in noise_levels
                        if per_noise_results[str(ns)]['advantage_significant'])

    print(f"\n=== Summary ===")
    print(f"Significant advantage at {n_significant}/{len(noise_levels)} noise levels")
    print(f"Spearman(noise, advantage): rho={rho:.4f}, p={p_corr:.4e}")

    results = {
        'experiment': 'noise_robustness',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p, 'rank': args.rank,
            'n_steps': args.n_steps, 'lr': args.lr,
            'noise_levels': noise_levels,
            'n_seeds': args.n_seeds, 'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'per_noise_level': per_noise_results,
        'robustness_summary': {
            'n_significant': n_significant,
            'n_noise_levels': len(noise_levels),
            'spearman_noise_vs_advantage': float(rho),
            'spearman_p': float(p_corr),
            'advantage_at_zero_noise': float(per_noise_results['0.0']['H_advantage_mean']),
            'advantage_at_max_noise': float(per_noise_results[str(noise_levels[-1])]['H_advantage_mean']),
        },
    }

    save_results(results, args.output_dir, '15_noise_robustness.json')


if __name__ == '__main__':
    main()
