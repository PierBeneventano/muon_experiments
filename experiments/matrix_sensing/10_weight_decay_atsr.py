#!/usr/bin/env python3
"""Muon with weight decay: ATSR degradation.

Muon with weight decay lambda in {0, 1e-4, 1e-3, 1e-2}.
K=4 blocks, m0=5, p=200, 50000 steps.
Measure ATSR degradation. 10 seeds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.optimizers import muon_step, muon_wd_step
from src.metrics import spectral_entropy, max_entropy, block_singular_value_mass, atsr
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


def compute_acquisition_times(mass_snapshots, K, threshold=0.5):
    """Find acquisition times from mass trajectory."""
    target_mass = threshold / K
    acq_times = [0] * K
    for k in range(K):
        for step, masses in mass_snapshots:
            if masses[k] >= target_mass:
                acq_times[k] = step
                break
    return acq_times


def main():
    parser = get_parser("Weight decay ATSR degradation")
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--m0', type=int, default=5)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--track_interval', type=int, default=200)
    args = parser.parse_args()
    args.n_seeds = 10

    m = args.K * args.m0
    H_max = max_entropy(m, m)
    weight_decays = [0.0, 1e-4, 1e-3, 1e-2]

    per_wd_results = {}

    for wd in weight_decays:
        print(f"\n=== weight_decay={wd} ===")
        atsr_values = []
        H_values = []

        for s_idx in range(args.n_seeds):
            seed = args.seed + s_idx
            W_star, A, b, block_bounds = generate_block_problem(args.K, args.m0, args.p, seed)

            set_seed(seed + 10000)
            W = np.random.randn(m, m) * 0.01

            mass_snapshots = []
            for t in range(args.n_steps):
                if t % args.track_interval == 0:
                    masses = block_singular_value_mass(W, block_bounds)
                    mass_snapshots.append((t, masses))

                _, G = compute_loss_and_gradient(W, A, b)

                if wd > 0:
                    W = muon_wd_step(W, G, args.lr, weight_decay=wd)
                else:
                    W, _ = muon_step(W, G, args.lr)

            # Final snapshot
            masses = block_singular_value_mass(W, block_bounds)
            mass_snapshots.append((args.n_steps, masses))

            acq_times = compute_acquisition_times(mass_snapshots, args.K)
            atsr_val = atsr(acq_times)
            H_final = spectral_entropy(W)

            atsr_values.append(float(atsr_val))
            H_values.append(float(H_final))

            print(f"  Seed {s_idx+1}: ATSR={atsr_val:.2f}, H/H_max={H_final/H_max:.4f}")

        atsr_arr = np.array(atsr_values)
        H_arr = np.array(H_values)
        per_wd_results[str(wd)] = {
            'weight_decay': wd,
            'atsr_values': atsr_values,
            'H_values': H_values,
            'atsr_mean': float(np.mean(atsr_arr)),
            'atsr_std': float(np.std(atsr_arr, ddof=1)),
            'H_mean': float(np.mean(H_arr)),
            'H_over_Hmax_mean': float(np.mean(H_arr / H_max)),
        }
        print(f"  ATSR = {np.mean(atsr_arr):.2f} +/- {np.std(atsr_arr, ddof=1):.2f}")
        print(f"  H/H_max = {np.mean(H_arr)/H_max:.4f}")

    # Compare: ATSR increase from wd=0 baseline
    baseline_atsr = np.mean(per_wd_results['0.0']['atsr_values'])
    print(f"\n=== ATSR relative to baseline (wd=0: {baseline_atsr:.2f}) ===")
    for wd in weight_decays:
        mean_atsr = per_wd_results[str(wd)]['atsr_mean']
        ratio = mean_atsr / baseline_atsr if baseline_atsr > 0 else float('inf')
        print(f"  wd={wd}: ATSR={mean_atsr:.2f} (ratio={ratio:.2f}x)")
        per_wd_results[str(wd)]['atsr_ratio_to_baseline'] = float(ratio)

    results = {
        'experiment': 'weight_decay_atsr',
        'config': {
            'K': args.K, 'm0': args.m0, 'm': m, 'p': args.p,
            'n_steps': args.n_steps, 'lr': args.lr,
            'weight_decays': weight_decays,
            'n_seeds': args.n_seeds, 'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'per_weight_decay': per_wd_results,
    }

    save_results(results, args.output_dir, '10_weight_decay_atsr.json')


if __name__ == '__main__':
    main()
