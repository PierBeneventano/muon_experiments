#!/usr/bin/env python3
"""Kappa-controlled experiment.

m=n=20, p=300, rank=10, 8000 steps.
Kappa in {1.5, 2, 3, 5, 10, 20, 30}.
Muon and GD, 10 seeds each.
Measure H_advantage = H_muon - H_gd per kappa.
Report Spearman correlation and regression.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats

from src.optimizers import muon_step, gd_step
from src.metrics import spectral_entropy, max_entropy
from src.matrix_sensing import generate_problem, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def run_optimizer(optimizer_name, W_init, A, b, n_steps, lr):
    """Run optimizer, return final W."""
    W = W_init.copy()
    mom_buf = None
    for t in range(n_steps):
        _, G = compute_loss_and_gradient(W, A, b)
        if optimizer_name == 'muon':
            W, mom_buf = muon_step(W, G, lr)
        elif optimizer_name == 'gd':
            W = gd_step(W, G, lr)
    return W


def main():
    parser = get_parser("Kappa-controlled scaling experiment")
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--p', type=int, default=300)
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    args.n_seeds = 10

    kappas = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
    H_max = max_entropy(args.m, args.n)

    all_results = {}
    H_advantages = []
    kappa_values_for_corr = []

    for kappa in kappas:
        print(f"\n=== kappa={kappa} ===")
        H_muon_list = []
        H_gd_list = []

        for s_idx in range(args.n_seeds):
            seed = args.seed + s_idx
            set_seed(seed)
            W_star, A, b = generate_problem(args.m, args.n, args.p, args.rank,
                                            kappa=kappa, seed=seed)

            for opt_name in ['muon', 'gd']:
                set_seed(seed + 10000)
                W_init = np.random.randn(args.m, args.n) * 0.1
                W_final = run_optimizer(opt_name, W_init, A, b, args.n_steps, args.lr)
                H = spectral_entropy(W_final)
                if opt_name == 'muon':
                    H_muon_list.append(float(H))
                else:
                    H_gd_list.append(float(H))

        H_muon_arr = np.array(H_muon_list)
        H_gd_arr = np.array(H_gd_list)
        H_adv = H_muon_arr - H_gd_arr

        all_results[str(kappa)] = {
            'H_muon': H_muon_list,
            'H_gd': H_gd_list,
            'H_advantage_mean': float(np.mean(H_adv)),
            'H_advantage_std': float(np.std(H_adv, ddof=1)),
            'H_muon_mean': float(np.mean(H_muon_arr)),
            'H_gd_mean': float(np.mean(H_gd_arr)),
        }

        H_advantages.append(float(np.mean(H_adv)))
        kappa_values_for_corr.append(kappa)

        print(f"  Muon H/H_max: {np.mean(H_muon_arr)/H_max:.4f} +/- {np.std(H_muon_arr, ddof=1)/H_max:.4f}")
        print(f"  GD   H/H_max: {np.mean(H_gd_arr)/H_max:.4f} +/- {np.std(H_gd_arr, ddof=1)/H_max:.4f}")
        print(f"  H_advantage:  {np.mean(H_adv):.4f} +/- {np.std(H_adv, ddof=1):.4f}")

    # Spearman correlation: kappa vs H_advantage
    kappa_arr = np.array(kappa_values_for_corr)
    hadv_arr = np.array(H_advantages)
    rho, p_spearman = stats.spearmanr(kappa_arr, hadv_arr)

    # Linear regression: log(kappa) vs H_advantage
    log_kappa = np.log(kappa_arr)
    slope, intercept, r_value, p_linreg, std_err = stats.linregress(log_kappa, hadv_arr)

    print(f"\n=== Scaling analysis ===")
    print(f"Spearman rho={rho:.4f}, p={p_spearman:.4e}")
    print(f"Regression: H_adv = {slope:.4f} * log(kappa) + {intercept:.4f}, R^2={r_value**2:.4f}")

    results = {
        'experiment': 'kappa_scaling',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p, 'rank': args.rank,
            'n_steps': args.n_steps, 'lr': args.lr, 'n_seeds': args.n_seeds,
            'kappas': kappas, 'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'per_kappa': all_results,
        'scaling_analysis': {
            'spearman_rho': float(rho),
            'spearman_p': float(p_spearman),
            'regression_slope': float(slope),
            'regression_intercept': float(intercept),
            'regression_R2': float(r_value**2),
            'regression_p': float(p_linreg),
            'regression_stderr': float(std_err),
        },
    }

    save_results(results, args.output_dir, '03_kappa_scaling.json')


if __name__ == '__main__':
    main()
