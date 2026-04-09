#!/usr/bin/env python3
"""E1 learning rate sweep.

Same as E1 but sweep LR per optimizer.
LRs: {0.0003, 0.001, 0.003, 0.01, 0.03, 0.1}.
Phase 1: 5 pilot seeds per LR.
Phase 2: 20 production seeds at best LR.
Report H at optimal LRs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
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
    parser = get_parser("E1 learning rate sweep")
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--n_pilot_seeds', type=int, default=5)
    parser.add_argument('--n_prod_seeds', type=int, default=20)
    args = parser.parse_args()

    lrs = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    optimizers = ['muon', 'gd', 'nm_gd', 'random_orth']
    optimizer_labels = {'muon': 'Muon', 'gd': 'GD', 'nm_gd': 'NM-GD', 'random_orth': 'Random-Orth'}
    H_max = max_entropy(args.m, args.n)

    # Phase 1: Pilot sweep
    print("=== Phase 1: Pilot sweep ===")
    pilot_results = {opt: {} for opt in optimizers}

    for opt in optimizers:
        for lr in lrs:
            H_pilot = []
            diverged = False
            for s_idx in range(args.n_pilot_seeds):
                seed = args.seed + s_idx
                set_seed(seed)
                W_star, A, b = generate_problem(args.m, args.n, args.p, args.rank, seed=seed)
                set_seed(seed + 10000)
                W_init = np.random.randn(args.m, args.n) * 0.1
                W_final = run_optimizer(opt, W_init, A, b, args.n_steps, lr)
                if np.any(np.isnan(W_final)) or np.any(np.isinf(W_final)):
                    diverged = True
                    break
                H_pilot.append(float(spectral_entropy(W_final)))

            if diverged:
                pilot_results[opt][lr] = {'mean_H': -1.0, 'diverged': True}
            else:
                pilot_results[opt][lr] = {
                    'mean_H': float(np.mean(H_pilot)),
                    'diverged': False,
                    'H_values': H_pilot,
                }
            status = "DIVERGED" if diverged else f"H={pilot_results[opt][lr]['mean_H']:.4f}"
            print(f"  {optimizer_labels[opt]} lr={lr}: {status}")

    # Select best LR per optimizer (highest mean H among non-diverged)
    best_lrs = {}
    for opt in optimizers:
        valid = {lr: v['mean_H'] for lr, v in pilot_results[opt].items() if not v['diverged']}
        if valid:
            best_lrs[opt] = max(valid, key=valid.get)
        else:
            best_lrs[opt] = lrs[3]  # fallback to 0.01
        print(f"  Best LR for {optimizer_labels[opt]}: {best_lrs[opt]}")

    # Phase 2: Production runs at best LR
    print("\n=== Phase 2: Production runs ===")
    prod_results = {opt: [] for opt in optimizers}

    for opt in optimizers:
        lr = best_lrs[opt]
        for s_idx in range(args.n_prod_seeds):
            seed = args.seed + s_idx
            set_seed(seed)
            W_star, A, b = generate_problem(args.m, args.n, args.p, args.rank, seed=seed)
            set_seed(seed + 10000)
            W_init = np.random.randn(args.m, args.n) * 0.1
            W_final = run_optimizer(opt, W_init, A, b, args.n_steps, lr)
            H = spectral_entropy(W_final)
            prod_results[opt].append(float(H))
        mean_H = np.mean(prod_results[opt])
        print(f"  {optimizer_labels[opt]} (lr={lr}): H/H_max = {mean_H/H_max:.4f}")

    results = {
        'experiment': 'E1_lr_sweep',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p, 'rank': args.rank,
            'n_steps': args.n_steps, 'lrs': lrs,
            'n_pilot_seeds': args.n_pilot_seeds,
            'n_prod_seeds': args.n_prod_seeds,
            'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'pilot_results': {
            optimizer_labels[opt]: {str(lr): v for lr, v in pilot_results[opt].items()}
            for opt in optimizers
        },
        'best_lrs': {optimizer_labels[opt]: best_lrs[opt] for opt in optimizers},
        'production_H_values': {optimizer_labels[opt]: prod_results[opt] for opt in optimizers},
        'production_summary': {
            optimizer_labels[opt]: {
                'best_lr': best_lrs[opt],
                'H_mean': float(np.mean(prod_results[opt])),
                'H_std': float(np.std(prod_results[opt], ddof=1)),
                'H_over_Hmax_mean': float(np.mean(np.array(prod_results[opt]) / H_max)),
            }
            for opt in optimizers
        },
    }

    save_results(results, args.output_dir, '02_e1_lr_sweep.json')


if __name__ == '__main__':
    main()
