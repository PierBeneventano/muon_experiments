#!/usr/bin/env python3
"""Block-diagonal matrix sensing: block acquisition dynamics.

K=4 blocks, m0=5 (so m=20, n=20), p=200.
50000 steps. Muon / GD / NM-GD / Adam. 10 seeds.
Measure ATSR (acquisition time spread ratio).
Report block acquisition trajectories.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.optimizers import muon_step, gd_step, norm_matched_gd_step, adam_step
from src.metrics import spectral_entropy, nuclear_norm, block_singular_value_mass, atsr
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


def run_with_tracking(optimizer_name, W_init, A, b, block_bounds, n_steps, lr,
                      track_interval=100):
    """Run optimizer and track block acquisition over time."""
    W = W_init.copy()
    mom_buf = None
    m_buf = None
    v_buf = None
    K = len(block_bounds)

    trajectory = []  # list of (step, block_masses)

    for t in range(n_steps):
        if t % track_interval == 0:
            masses = block_singular_value_mass(W, block_bounds)
            trajectory.append({'step': t, 'block_masses': [float(x) for x in masses]})

        _, G = compute_loss_and_gradient(W, A, b)

        if optimizer_name == 'muon':
            W, mom_buf = muon_step(W, G, lr)
        elif optimizer_name == 'gd':
            W = gd_step(W, G, lr)
        elif optimizer_name == 'nm_gd':
            W = norm_matched_gd_step(W, G, lr)
        elif optimizer_name == 'adam':
            W, m_buf, v_buf = adam_step(W, G, lr, t, m_buf, v_buf)

    # Final masses
    masses = block_singular_value_mass(W, block_bounds)
    trajectory.append({'step': n_steps, 'block_masses': [float(x) for x in masses]})

    return W, trajectory


def compute_acquisition_times(trajectory, K, threshold=0.5):
    """
    For each block, find the first step where block_mass >= threshold * (1/K).
    threshold=0.5 means half of the equal-share mass.
    """
    target_mass = threshold / K
    acq_times = [0] * K
    for k in range(K):
        for snap in trajectory:
            if snap['block_masses'][k] >= target_mass:
                acq_times[k] = snap['step']
                break
    return acq_times


def main():
    parser = get_parser("Block-diagonal acquisition dynamics")
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--m0', type=int, default=5)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--track_interval', type=int, default=200)
    args = parser.parse_args()
    args.n_seeds = 10

    m = args.K * args.m0
    optimizers = ['muon', 'gd', 'nm_gd', 'adam']
    optimizer_labels = {'muon': 'Muon', 'gd': 'GD', 'nm_gd': 'NM-GD', 'adam': 'Adam'}

    all_atsr = {opt: [] for opt in optimizers}
    all_trajectories = {opt: [] for opt in optimizers}

    for s_idx in range(args.n_seeds):
        seed = args.seed + s_idx
        print(f"Seed {s_idx+1}/{args.n_seeds} (seed={seed})")

        W_star, A, b, block_bounds = generate_block_problem(args.K, args.m0, args.p, seed)

        for opt in optimizers:
            set_seed(seed + 10000)
            W_init = np.random.randn(m, m) * 0.01

            W_final, traj = run_with_tracking(
                opt, W_init, A, b, block_bounds, args.n_steps, args.lr,
                track_interval=args.track_interval)

            acq_times = compute_acquisition_times(traj, args.K)
            atsr_val = atsr(acq_times)
            all_atsr[opt].append(float(atsr_val))
            all_trajectories[opt].append({
                'seed': seed,
                'acquisition_times': acq_times,
                'atsr': float(atsr_val),
                'final_H': float(spectral_entropy(W_final)),
                # Store subsampled trajectory (every 10th snapshot)
                'trajectory': traj[::10],
            })
            print(f"  {optimizer_labels[opt]}: ATSR={atsr_val:.2f}, "
                  f"acq_times={acq_times}")

    # Summary
    summary = {}
    for opt in optimizers:
        arr = np.array(all_atsr[opt])
        summary[optimizer_labels[opt]] = {
            'atsr_mean': float(np.mean(arr)),
            'atsr_std': float(np.std(arr, ddof=1)),
            'atsr_median': float(np.median(arr)),
        }
        print(f"\n{optimizer_labels[opt]}: ATSR = {np.mean(arr):.2f} +/- {np.std(arr, ddof=1):.2f}")

    results = {
        'experiment': 'block_acquisition',
        'config': {
            'K': args.K, 'm0': args.m0, 'm': m, 'n': m, 'p': args.p,
            'n_steps': args.n_steps, 'lr': args.lr,
            'n_seeds': args.n_seeds, 'base_seed': args.seed,
            'track_interval': args.track_interval,
        },
        'summary': summary,
        'atsr_values': {optimizer_labels[opt]: all_atsr[opt] for opt in optimizers},
        'detailed': {optimizer_labels[opt]: all_trajectories[opt] for opt in optimizers},
    }

    save_results(results, args.output_dir, '05_block_acquisition.json')


if __name__ == '__main__':
    main()
