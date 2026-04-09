#!/usr/bin/env python3
"""Dimension scaling experiment.

n in {10, 20, 30, 50}. E1 setup at each dimension.
Muon and GD, 10 seeds. Does the H advantage grow with n?
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
        else:
            W = gd_step(W, G, lr)
    return W


def main():
    parser = get_parser("Dimension scaling: H advantage vs matrix dimension n")
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    args.n_seeds = 10

    dims = [10, 20, 30, 50]
    per_dim_results = {}
    advantages = []
    normalized_advantages = []

    for n_dim in dims:
        m_dim = n_dim
        p = 10 * n_dim  # scale measurements with dimension
        H_max = max_entropy(m_dim, n_dim)
        print(f"\n=== n={n_dim}, m={m_dim}, p={p} ===")

        H_muon = []
        H_gd = []

        for s_idx in range(args.n_seeds):
            seed = args.seed + s_idx
            set_seed(seed)
            W_star, A, b = generate_problem(m_dim, n_dim, p, args.rank, seed=seed)

            for opt in ['muon', 'gd']:
                set_seed(seed + 10000)
                W_init = np.random.randn(m_dim, n_dim) * 0.1
                W_final = run_optimizer(opt, W_init, A, b, args.n_steps, args.lr)
                H = spectral_entropy(W_final)
                if opt == 'muon':
                    H_muon.append(float(H))
                else:
                    H_gd.append(float(H))

        muon_arr = np.array(H_muon)
        gd_arr = np.array(H_gd)
        adv = muon_arr - gd_arr
        norm_adv = adv / H_max  # normalized advantage

        per_dim_results[n_dim] = {
            'n': n_dim, 'm': m_dim, 'p': p,
            'H_max': float(H_max),
            'H_muon': H_muon,
            'H_gd': H_gd,
            'advantage_mean': float(np.mean(adv)),
            'advantage_std': float(np.std(adv, ddof=1)),
            'normalized_advantage_mean': float(np.mean(norm_adv)),
            'normalized_advantage_std': float(np.std(norm_adv, ddof=1)),
            'muon_H_over_Hmax': float(np.mean(muon_arr / H_max)),
            'gd_H_over_Hmax': float(np.mean(gd_arr / H_max)),
        }

        advantages.append(float(np.mean(adv)))
        normalized_advantages.append(float(np.mean(norm_adv)))

        print(f"  Muon H/H_max: {np.mean(muon_arr)/H_max:.4f}")
        print(f"  GD   H/H_max: {np.mean(gd_arr)/H_max:.4f}")
        print(f"  Advantage:    {np.mean(adv):.4f} (normalized: {np.mean(norm_adv):.4f})")

    # Scaling analysis
    dims_arr = np.array(dims, dtype=float)
    adv_arr = np.array(advantages)
    nadv_arr = np.array(normalized_advantages)

    # Does raw advantage grow with n?
    rho_raw, p_raw = stats.spearmanr(dims_arr, adv_arr)
    slope_raw, intercept_raw, r_raw, p_linreg_raw, _ = stats.linregress(dims_arr, adv_arr)

    # Does normalized advantage grow with n?
    rho_norm, p_norm = stats.spearmanr(dims_arr, nadv_arr)
    slope_norm, intercept_norm, r_norm, p_linreg_norm, _ = stats.linregress(dims_arr, nadv_arr)

    print(f"\n=== Scaling analysis ===")
    print(f"Raw advantage vs n: Spearman rho={rho_raw:.4f} (p={p_raw:.4e})")
    print(f"  Linear: slope={slope_raw:.6f}, R^2={r_raw**2:.4f}")
    print(f"Normalized advantage vs n: Spearman rho={rho_norm:.4f} (p={p_norm:.4e})")
    print(f"  Linear: slope={slope_norm:.6f}, R^2={r_norm**2:.4f}")

    results = {
        'experiment': 'dimension_scaling',
        'config': {
            'dims': dims, 'rank': args.rank,
            'n_steps': args.n_steps, 'lr': args.lr,
            'n_seeds': args.n_seeds, 'base_seed': args.seed,
        },
        'per_dimension': {str(d): per_dim_results[d] for d in dims},
        'scaling_analysis': {
            'raw_advantage': {
                'spearman_rho': float(rho_raw),
                'spearman_p': float(p_raw),
                'linear_slope': float(slope_raw),
                'linear_R2': float(r_raw**2),
            },
            'normalized_advantage': {
                'spearman_rho': float(rho_norm),
                'spearman_p': float(p_norm),
                'linear_slope': float(slope_norm),
                'linear_R2': float(r_norm**2),
            },
        },
    }

    save_results(results, args.output_dir, '14_dimension_scaling.json')


if __name__ == '__main__':
    main()
