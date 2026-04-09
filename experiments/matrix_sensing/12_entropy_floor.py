#!/usr/bin/env python3
"""Entropy floor test.

Track H step-by-step. 3 initializations:
  isotropic, skewed (kappa=10), near-singular.
Muon, 1000 steps.
Check if H ever drops below 0.90*H_max after reaching it.
5 seeds per init.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy.linalg import svd

from src.optimizers import muon_step
from src.metrics import spectral_entropy, max_entropy
from src.matrix_sensing import generate_problem, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def make_init(init_type, m, n, seed):
    """Generate initial W based on init type."""
    set_seed(seed)
    if init_type == 'isotropic':
        # Random Gaussian, roughly isotropic singular values
        W = np.random.randn(m, n) * 0.1
    elif init_type == 'skewed':
        # High condition number initialization
        U = np.linalg.qr(np.random.randn(m, m))[0]
        V = np.linalg.qr(np.random.randn(n, n))[0]
        k = min(m, n)
        sigmas = np.linspace(10.0, 1.0, k)  # kappa=10
        W = U[:, :k] @ np.diag(sigmas * 0.01) @ V[:, :k].T
    elif init_type == 'near_singular':
        # Nearly rank-1
        u = np.random.randn(m, 1)
        v = np.random.randn(1, n)
        W = 0.1 * u @ v + 0.001 * np.random.randn(m, n)
    else:
        raise ValueError(f"Unknown init type: {init_type}")
    return W


def main():
    parser = get_parser("Entropy floor test: does H stay above 0.90*H_max?")
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--floor_threshold', type=float, default=0.90)
    args = parser.parse_args()
    args.n_seeds = 5

    H_max = max_entropy(args.m, args.n)
    floor_level = args.floor_threshold * H_max
    init_types = ['isotropic', 'skewed', 'near_singular']

    all_results = {}

    for init_type in init_types:
        print(f"\n=== Init: {init_type} ===")
        init_data = []

        for s_idx in range(args.n_seeds):
            seed = args.seed + s_idx

            # Generate problem
            set_seed(seed)
            W_star, A, b = generate_problem(args.m, args.n, args.p, args.rank, seed=seed)

            # Initialize W
            W = make_init(init_type, args.m, args.n, seed + 20000)
            mom_buf = None

            H_trace = []
            reached_floor = False
            first_reach_step = -1
            violations = []

            for t in range(args.n_steps):
                H = spectral_entropy(W)
                H_trace.append(float(H))

                # Track when we first reach the floor
                if not reached_floor and H >= floor_level:
                    reached_floor = True
                    first_reach_step = t

                # Check for violations after first reaching floor
                if reached_floor and H < floor_level:
                    violations.append({
                        'step': t,
                        'H': float(H),
                        'H_over_Hmax': float(H / H_max),
                    })

                _, G = compute_loss_and_gradient(W, A, b)
                W, mom_buf = muon_step(W, G, args.lr)

            # Final H
            H_final = spectral_entropy(W)
            H_trace.append(float(H_final))

            record = {
                'seed': seed,
                'init_type': init_type,
                'reached_floor': reached_floor,
                'first_reach_step': first_reach_step,
                'n_violations': len(violations),
                'violations': violations,
                'H_init': float(H_trace[0]),
                'H_final': float(H_trace[-1]),
                'H_min_after_reach': float(min(H_trace[first_reach_step:]) if reached_floor else 0.0),
                'H_trace': H_trace[::10],  # subsample for storage
            }
            init_data.append(record)

            status = "PASS" if len(violations) == 0 else f"FAIL ({len(violations)} violations)"
            if not reached_floor:
                status = "NEVER REACHED"
            print(f"  Seed {s_idx+1}: {status}, "
                  f"H_init={H_trace[0]/H_max:.4f}, H_final={H_final/H_max:.4f}")

        # Summary for this init type
        n_passed = sum(1 for r in init_data if r['reached_floor'] and r['n_violations'] == 0)
        n_reached = sum(1 for r in init_data if r['reached_floor'])
        all_results[init_type] = {
            'per_seed': init_data,
            'n_reached_floor': n_reached,
            'n_passed': n_passed,
            'n_seeds': args.n_seeds,
            'pass_rate': n_passed / args.n_seeds,
        }
        print(f"  Floor test: {n_passed}/{args.n_seeds} passed "
              f"({n_reached}/{args.n_seeds} reached floor)")

    results = {
        'experiment': 'entropy_floor',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p, 'rank': args.rank,
            'n_steps': args.n_steps, 'lr': args.lr,
            'floor_threshold': args.floor_threshold,
            'n_seeds': args.n_seeds, 'base_seed': args.seed,
            'init_types': init_types,
        },
        'H_max': float(H_max),
        'floor_level': float(floor_level),
        'per_init_type': all_results,
        'overall_pass_rate': float(np.mean([
            all_results[it]['pass_rate'] for it in init_types
        ])),
    }

    save_results(results, args.output_dir, '12_entropy_floor.json')


if __name__ == '__main__':
    main()
