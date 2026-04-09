#!/usr/bin/env python3
"""Track principal angles between W and G singular frames during Muon training.

m=n=20, p=200, rank=2, 3000 steps. 5 seeds.
Report per-step angles and entropy changes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy.linalg import svd

from src.optimizers import muon_step
from src.metrics import spectral_entropy, principal_angles, max_entropy
from src.matrix_sensing import generate_problem, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def main():
    parser = get_parser("Alignment tracking: principal angles between W and G frames")
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--track_interval', type=int, default=10)
    args = parser.parse_args()
    args.n_seeds = 5

    H_max = max_entropy(args.m, args.n)
    all_seed_data = []

    for s_idx in range(args.n_seeds):
        seed = args.seed + s_idx
        print(f"Seed {s_idx+1}/{args.n_seeds} (seed={seed})")

        set_seed(seed)
        W_star, A, b = generate_problem(args.m, args.n, args.p, args.rank, seed=seed)

        set_seed(seed + 10000)
        W = np.random.randn(args.m, args.n) * 0.1
        mom_buf = None

        tracking = []

        for t in range(args.n_steps):
            loss, G = compute_loss_and_gradient(W, A, b)

            if t % args.track_interval == 0:
                H = spectral_entropy(W)

                # SVD of W and G
                U_W, s_W, Vt_W = svd(W, full_matrices=False)
                U_G, s_G, Vt_G = svd(G, full_matrices=False)

                # Principal angles between left singular frames (top-k)
                k = min(args.rank, args.m, args.n)
                angles_left = principal_angles(U_W[:, :k], U_G[:, :k])
                angles_right = principal_angles(Vt_W[:k, :].T, Vt_G[:k, :].T)

                # Mean cosine of principal angles (alignment measure)
                mean_cos_left = float(np.mean(np.cos(angles_left)))
                mean_cos_right = float(np.mean(np.cos(angles_right)))

                tracking.append({
                    'step': t,
                    'loss': float(loss),
                    'H': float(H),
                    'H_over_Hmax': float(H / H_max),
                    'angles_left_deg': [float(np.degrees(a)) for a in angles_left],
                    'angles_right_deg': [float(np.degrees(a)) for a in angles_right],
                    'mean_cos_left': mean_cos_left,
                    'mean_cos_right': mean_cos_right,
                    'sigma_W_top': [float(x) for x in s_W[:k]],
                    'sigma_G_top': [float(x) for x in s_G[:k]],
                    'grad_fro_norm': float(np.linalg.norm(G, 'fro')),
                })

            W, mom_buf = muon_step(W, G, args.lr)

        # Compute entropy changes
        for i in range(1, len(tracking)):
            tracking[i]['dH'] = tracking[i]['H'] - tracking[i-1]['H']
        tracking[0]['dH'] = 0.0

        all_seed_data.append({
            'seed': seed,
            'final_H': float(spectral_entropy(W)),
            'final_H_over_Hmax': float(spectral_entropy(W) / H_max),
            'tracking': tracking,
        })

        print(f"  Final H/H_max = {spectral_entropy(W)/H_max:.4f}")

    # Aggregate: mean alignment over seeds at each tracked step
    # Find common steps
    steps = [t['step'] for t in all_seed_data[0]['tracking']]
    agg_alignment = []
    for step_idx, step in enumerate(steps):
        cos_lefts = [sd['tracking'][step_idx]['mean_cos_left'] for sd in all_seed_data
                     if step_idx < len(sd['tracking'])]
        cos_rights = [sd['tracking'][step_idx]['mean_cos_right'] for sd in all_seed_data
                      if step_idx < len(sd['tracking'])]
        Hs = [sd['tracking'][step_idx]['H'] for sd in all_seed_data
              if step_idx < len(sd['tracking'])]
        agg_alignment.append({
            'step': step,
            'mean_cos_left': float(np.mean(cos_lefts)),
            'mean_cos_right': float(np.mean(cos_rights)),
            'mean_H': float(np.mean(Hs)),
        })

    results = {
        'experiment': 'alignment_tracking',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p, 'rank': args.rank,
            'n_steps': args.n_steps, 'lr': args.lr,
            'n_seeds': args.n_seeds, 'track_interval': args.track_interval,
            'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'per_seed': all_seed_data,
        'aggregated_alignment': agg_alignment,
    }

    save_results(results, args.output_dir, '08_alignment_tracking.json')


if __name__ == '__main__':
    main()
