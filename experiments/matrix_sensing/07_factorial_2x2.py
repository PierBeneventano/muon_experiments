#!/usr/bin/env python3
"""2x2 factorial design: polar vs norm contributions.

K=4 block-diagonal. Four optimizers:
  Muon       = polar + norm  (polar map, all SVs = 1)
  GD         = no polar + no norm  (raw gradient)
  NM-GD      = no polar + norm  (gradient rescaled to Muon's norm)
  Polar-unnorm = polar + no norm  (polar direction, gradient's magnitude)

Measure H and ATSR.
Decompose into isometry, alignment, magnitude contributions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.optimizers import muon_step, gd_step, norm_matched_gd_step, polar_unnormalized_step
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


def run_with_tracking(optimizer_name, W_init, A, b, block_bounds, n_steps, lr,
                      track_interval=200):
    """Run optimizer and track metrics."""
    W = W_init.copy()
    mom_buf = None
    K = len(block_bounds)

    H_trace = []
    mass_trace = []

    for t in range(n_steps):
        if t % track_interval == 0:
            H_trace.append({'step': t, 'H': float(spectral_entropy(W))})
            masses = block_singular_value_mass(W, block_bounds)
            mass_trace.append({'step': t, 'masses': [float(x) for x in masses]})

        _, G = compute_loss_and_gradient(W, A, b)

        if optimizer_name == 'muon':
            W, mom_buf = muon_step(W, G, lr)
        elif optimizer_name == 'gd':
            W = gd_step(W, G, lr)
        elif optimizer_name == 'nm_gd':
            W = norm_matched_gd_step(W, G, lr)
        elif optimizer_name == 'polar_unnorm':
            W = polar_unnormalized_step(W, G, lr)

    # Final metrics
    H_final = spectral_entropy(W)
    H_trace.append({'step': n_steps, 'H': float(H_final)})
    masses = block_singular_value_mass(W, block_bounds)
    mass_trace.append({'step': n_steps, 'masses': [float(x) for x in masses]})

    return W, H_trace, mass_trace


def compute_acquisition_times(mass_trace, K, threshold=0.5):
    """Find acquisition times from mass trajectory."""
    target_mass = threshold / K
    acq_times = [0] * K
    for k in range(K):
        for snap in mass_trace:
            if snap['masses'][k] >= target_mass:
                acq_times[k] = snap['step']
                break
    return acq_times


def main():
    parser = get_parser("2x2 factorial: polar x norm contributions")
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--m0', type=int, default=5)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    args.n_seeds = 10

    m = args.K * args.m0
    H_max = max_entropy(m, m)

    # 2x2 factorial: (polar, norm)
    # Muon:         (yes, yes)
    # GD:           (no,  no)
    # NM-GD:        (no,  yes)
    # Polar-unnorm: (yes, no)
    optimizers = ['muon', 'gd', 'nm_gd', 'polar_unnorm']
    labels = {
        'muon': 'Muon (polar+norm)',
        'gd': 'GD (raw)',
        'nm_gd': 'NM-GD (norm only)',
        'polar_unnorm': 'Polar-unnorm (polar only)',
    }
    factorial = {
        'muon': {'polar': True, 'norm': True},
        'gd': {'polar': False, 'norm': False},
        'nm_gd': {'polar': False, 'norm': True},
        'polar_unnorm': {'polar': True, 'norm': False},
    }

    all_H = {opt: [] for opt in optimizers}
    all_atsr = {opt: [] for opt in optimizers}

    for s_idx in range(args.n_seeds):
        seed = args.seed + s_idx
        print(f"Seed {s_idx+1}/{args.n_seeds} (seed={seed})")

        W_star, A, b, block_bounds = generate_block_problem(args.K, args.m0, args.p, seed)

        for opt in optimizers:
            set_seed(seed + 10000)
            W_init = np.random.randn(m, m) * 0.01
            W_final, H_trace, mass_trace = run_with_tracking(
                opt, W_init, A, b, block_bounds, args.n_steps, args.lr)

            H_final = spectral_entropy(W_final)
            acq_times = compute_acquisition_times(mass_trace, args.K)
            atsr_val = atsr(acq_times)

            all_H[opt].append(float(H_final))
            all_atsr[opt].append(float(atsr_val))

            print(f"  {labels[opt]}: H/H_max={H_final/H_max:.4f}, ATSR={atsr_val:.2f}")

    # Factorial decomposition (ANOVA-style)
    # Main effect of polar: mean(polar=yes) - mean(polar=no)
    polar_yes_H = np.array(all_H['muon'] + all_H['polar_unnorm'])
    polar_no_H = np.array(all_H['gd'] + all_H['nm_gd'])
    norm_yes_H = np.array(all_H['muon'] + all_H['nm_gd'])
    norm_no_H = np.array(all_H['gd'] + all_H['polar_unnorm'])

    polar_effect_H = float(np.mean(polar_yes_H) - np.mean(polar_no_H))
    norm_effect_H = float(np.mean(norm_yes_H) - np.mean(norm_no_H))

    # Interaction: (muon - nm_gd) - (polar_unnorm - gd)
    interaction_H = float(
        (np.mean(all_H['muon']) - np.mean(all_H['nm_gd'])) -
        (np.mean(all_H['polar_unnorm']) - np.mean(all_H['gd']))
    )

    # Same for ATSR
    polar_yes_atsr = np.array(all_atsr['muon'] + all_atsr['polar_unnorm'])
    polar_no_atsr = np.array(all_atsr['gd'] + all_atsr['nm_gd'])
    norm_yes_atsr = np.array(all_atsr['muon'] + all_atsr['nm_gd'])
    norm_no_atsr = np.array(all_atsr['gd'] + all_atsr['polar_unnorm'])

    polar_effect_atsr = float(np.mean(polar_yes_atsr) - np.mean(polar_no_atsr))
    norm_effect_atsr = float(np.mean(norm_yes_atsr) - np.mean(norm_no_atsr))
    interaction_atsr = float(
        (np.mean(all_atsr['muon']) - np.mean(all_atsr['nm_gd'])) -
        (np.mean(all_atsr['polar_unnorm']) - np.mean(all_atsr['gd']))
    )

    print(f"\n=== Factorial Decomposition ===")
    print(f"Entropy H:")
    print(f"  Polar main effect: {polar_effect_H:.4f}")
    print(f"  Norm main effect:  {norm_effect_H:.4f}")
    print(f"  Interaction:       {interaction_H:.4f}")
    print(f"ATSR:")
    print(f"  Polar main effect: {polar_effect_atsr:.4f}")
    print(f"  Norm main effect:  {norm_effect_atsr:.4f}")
    print(f"  Interaction:       {interaction_atsr:.4f}")

    results = {
        'experiment': 'factorial_2x2',
        'config': {
            'K': args.K, 'm0': args.m0, 'm': m, 'p': args.p,
            'n_steps': args.n_steps, 'lr': args.lr,
            'n_seeds': args.n_seeds, 'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'factorial_design': {labels[opt]: factorial[opt] for opt in optimizers},
        'H_values': {labels[opt]: all_H[opt] for opt in optimizers},
        'atsr_values': {labels[opt]: all_atsr[opt] for opt in optimizers},
        'summary': {
            labels[opt]: {
                'H_mean': float(np.mean(all_H[opt])),
                'H_std': float(np.std(all_H[opt], ddof=1)),
                'H_over_Hmax': float(np.mean(all_H[opt]) / H_max),
                'atsr_mean': float(np.mean(all_atsr[opt])),
                'atsr_std': float(np.std(all_atsr[opt], ddof=1)),
            }
            for opt in optimizers
        },
        'factorial_decomposition': {
            'H': {
                'polar_main_effect': polar_effect_H,
                'norm_main_effect': norm_effect_H,
                'interaction': interaction_H,
            },
            'ATSR': {
                'polar_main_effect': polar_effect_atsr,
                'norm_main_effect': norm_effect_atsr,
                'interaction': interaction_atsr,
            },
        },
    }

    save_results(results, args.output_dir, '07_factorial_2x2.json')


if __name__ == '__main__':
    main()
