#!/usr/bin/env python3
"""Block-diagonal K sweep.

K in {2, 4, 6, 8, 12}. m0=5, 50000 steps.
Muon only, 10 seeds.
Measure ATSR(K). Test if gamma(K) = O(log K).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats

from src.optimizers import muon_step
from src.metrics import spectral_entropy, block_singular_value_mass, atsr
from src.matrix_sensing import generate_block_diagonal_target, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def generate_block_problem(K, m0, p, seed):
    """Generate block-diagonal matrix sensing problem."""
    set_seed(seed)
    W_star, block_bounds = generate_block_diagonal_target(K, m0)
    m = K * m0
    A = np.random.randn(p, m, m) / np.sqrt(p)
    b = np.array([np.sum(A[i] * W_star) for i in range(p)])
    return W_star, A, b, block_bounds


def compute_acquisition_times(W_history, block_bounds, K, threshold=0.5):
    """Find acquisition times from stored W snapshots."""
    target_mass = threshold / K
    acq_times = [0] * K
    for k in range(K):
        for step, W in W_history:
            masses = block_singular_value_mass(W, block_bounds)
            if masses[k] >= target_mass:
                acq_times[k] = step
                break
    return acq_times


def main():
    parser = get_parser("Block-diagonal K sweep: ATSR vs K")
    parser.add_argument('--m0', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--track_interval', type=int, default=200)
    args = parser.parse_args()
    args.n_seeds = 10

    Ks = [2, 4, 6, 8, 12]
    per_K_results = {}
    mean_atsrs = []

    for K in Ks:
        m = K * args.m0
        # Scale p proportional to m^2 (enough measurements)
        p = max(200, 5 * m)
        print(f"\n=== K={K}, m={m}, p={p} ===")

        atsr_values = []
        for s_idx in range(args.n_seeds):
            seed = args.seed + s_idx
            W_star, A, b, block_bounds = generate_block_problem(K, args.m0, p, seed)

            set_seed(seed + 10000)
            W = np.random.randn(m, m) * 0.01
            mom_buf = None

            # Track block masses at intervals
            W_snapshots = []
            for t in range(args.n_steps):
                if t % args.track_interval == 0:
                    W_snapshots.append((t, W.copy()))
                _, G = compute_loss_and_gradient(W, A, b)
                W, mom_buf = muon_step(W, G, args.lr)

            W_snapshots.append((args.n_steps, W.copy()))

            acq_times = compute_acquisition_times(W_snapshots, block_bounds, K)
            atsr_val = atsr(acq_times)
            atsr_values.append(float(atsr_val))
            print(f"  Seed {s_idx+1}: ATSR={atsr_val:.2f}, acq_times={acq_times}")

        arr = np.array(atsr_values)
        per_K_results[K] = {
            'atsr_values': atsr_values,
            'atsr_mean': float(np.mean(arr)),
            'atsr_std': float(np.std(arr, ddof=1)),
            'atsr_median': float(np.median(arr)),
            'm': m, 'p': p,
        }
        mean_atsrs.append(float(np.mean(arr)))
        print(f"  K={K}: ATSR = {np.mean(arr):.2f} +/- {np.std(arr, ddof=1):.2f}")

    # Test gamma(K) = O(log K) by regressing ATSR on log(K)
    log_K = np.log(np.array(Ks, dtype=float))
    atsr_arr = np.array(mean_atsrs)
    slope, intercept, r_value, p_val, std_err = stats.linregress(log_K, atsr_arr)
    rho, p_spearman = stats.spearmanr(Ks, mean_atsrs)

    print(f"\n=== Scaling analysis ===")
    print(f"Regression: ATSR = {slope:.4f} * log(K) + {intercept:.4f}, R^2={r_value**2:.4f}")
    print(f"Spearman rho={rho:.4f}, p={p_spearman:.4e}")

    results = {
        'experiment': 'block_K_sweep',
        'config': {
            'Ks': Ks, 'm0': args.m0, 'n_steps': args.n_steps, 'lr': args.lr,
            'n_seeds': args.n_seeds, 'base_seed': args.seed,
        },
        'per_K': {str(K): per_K_results[K] for K in Ks},
        'scaling_analysis': {
            'log_K_regression_slope': float(slope),
            'log_K_regression_intercept': float(intercept),
            'log_K_regression_R2': float(r_value**2),
            'log_K_regression_p': float(p_val),
            'spearman_rho': float(rho),
            'spearman_p': float(p_spearman),
        },
    }

    save_results(results, args.output_dir, '06_block_k_sweep.json')


if __name__ == '__main__':
    main()
