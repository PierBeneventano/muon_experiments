#!/usr/bin/env python3
"""Compute full A(W,G) matrix at each step, extract Tr(A).

Compare Tr(A) to actual dH/deta.
m=n=10, p=100, 2000 steps.
Report within-phase Spearman correlation. 5 seeds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy.linalg import svd
from scipy import stats

from src.optimizers import muon_step, polar
from src.metrics import spectral_entropy, max_entropy
from src.matrix_sensing import generate_problem, compute_loss_and_gradient
from src.utils import set_seed, save_results, get_parser


def compute_A_matrix(W, G):
    """
    Compute the A(W,G) matrix that governs spectral dynamics under polar map.

    For W = U diag(sigma) V^T and P = polar(G) = U_G V_G^T,
    A_{ij} = (U^T U_G)_{ij}^2 * f(sigma_i, sigma_j)
    where f depends on the interaction between singular components.

    We compute the full matrix and return its trace.
    """
    U_W, s_W, Vt_W = svd(W, full_matrices=False)
    P = polar(G)
    U_P, s_P, Vt_P = svd(P, full_matrices=False)

    # Overlap matrices
    M_left = U_W.T @ U_P   # k x k
    M_right = Vt_W @ Vt_P.T  # k x k

    k = min(len(s_W), M_left.shape[1])
    M_left = M_left[:k, :k]
    M_right = M_right[:k, :k]

    # A matrix: A_{ij} = M_left_{ij} * M_right_{ij}
    # This captures how much of polar(G)'s i-th singular component
    # aligns with W's j-th singular component
    A = M_left * M_right

    return A, np.trace(A)


def compute_numerical_dH_deta(W, G, eta, eps_eta=1e-5):
    """
    Numerical derivative of H w.r.t. eta at the current step.
    dH/deta ~ (H(W - (eta+eps)*P) - H(W - (eta-eps)*P)) / (2*eps)
    """
    P = polar(G)
    W_plus = W - (eta + eps_eta) * P
    W_minus = W - (eta - eps_eta) * P
    H_plus = spectral_entropy(W_plus)
    H_minus = spectral_entropy(W_minus)
    return (H_plus - H_minus) / (2 * eps_eta)


def main():
    parser = get_parser("Exact Tr(A) computation and dH/deta correlation")
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--p', type=int, default=100)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--track_interval', type=int, default=5)
    args = parser.parse_args()
    args.n_seeds = 5

    H_max = max_entropy(args.m, args.n)
    all_seed_data = []

    for s_idx in range(args.n_seeds):
        seed = args.seed + s_idx
        print(f"Seed {s_idx+1}/{args.n_seeds} (seed={seed})")

        set_seed(seed)
        W_star, A_meas, b = generate_problem(args.m, args.n, args.p, args.rank, seed=seed)

        set_seed(seed + 10000)
        W = np.random.randn(args.m, args.n) * 0.1
        mom_buf = None

        tracking = []

        for t in range(args.n_steps):
            _, G = compute_loss_and_gradient(W, A_meas, b)

            if t % args.track_interval == 0:
                H = spectral_entropy(W)
                A_mat, tr_A = compute_A_matrix(W, G)
                dH_deta = compute_numerical_dH_deta(W, G, args.lr)

                tracking.append({
                    'step': t,
                    'H': float(H),
                    'H_over_Hmax': float(H / H_max),
                    'Tr_A': float(tr_A),
                    'dH_deta_numerical': float(dH_deta),
                    'A_frobenius': float(np.linalg.norm(A_mat, 'fro')),
                })

            W, mom_buf = muon_step(W, G, args.lr)

        # Compute within-phase correlations
        # Phase 1: early training (first 40%)
        # Phase 2: late training (last 40%)
        n_track = len(tracking)
        phase1_end = int(0.4 * n_track)
        phase2_start = int(0.6 * n_track)

        trA_all = [t['Tr_A'] for t in tracking]
        dHdeta_all = [t['dH_deta_numerical'] for t in tracking]

        trA_p1 = [t['Tr_A'] for t in tracking[:phase1_end]]
        dHdeta_p1 = [t['dH_deta_numerical'] for t in tracking[:phase1_end]]

        trA_p2 = [t['Tr_A'] for t in tracking[phase2_start:]]
        dHdeta_p2 = [t['dH_deta_numerical'] for t in tracking[phase2_start:]]

        def safe_spearman(x, y):
            if len(x) < 3:
                return 0.0, 1.0
            rho, p_val = stats.spearmanr(x, y)
            return float(rho), float(p_val)

        rho_all, p_all = safe_spearman(trA_all, dHdeta_all)
        rho_p1, p_p1 = safe_spearman(trA_p1, dHdeta_p1)
        rho_p2, p_p2 = safe_spearman(trA_p2, dHdeta_p2)

        seed_data = {
            'seed': seed,
            'tracking': tracking,
            'correlations': {
                'full': {'spearman_rho': rho_all, 'p_value': p_all},
                'phase1_early': {'spearman_rho': rho_p1, 'p_value': p_p1},
                'phase2_late': {'spearman_rho': rho_p2, 'p_value': p_p2},
            },
        }
        all_seed_data.append(seed_data)

        print(f"  Spearman(Tr(A), dH/deta): full={rho_all:.4f}, "
              f"early={rho_p1:.4f}, late={rho_p2:.4f}")

    # Aggregate correlations
    agg_corrs = {
        phase: {
            'mean_rho': float(np.mean([sd['correlations'][phase]['spearman_rho']
                                       for sd in all_seed_data])),
            'std_rho': float(np.std([sd['correlations'][phase]['spearman_rho']
                                     for sd in all_seed_data], ddof=1)),
        }
        for phase in ['full', 'phase1_early', 'phase2_late']
    }

    results = {
        'experiment': 'exact_trA',
        'config': {
            'm': args.m, 'n': args.n, 'p': args.p, 'rank': args.rank,
            'n_steps': args.n_steps, 'lr': args.lr,
            'n_seeds': args.n_seeds, 'track_interval': args.track_interval,
            'base_seed': args.seed,
        },
        'H_max': float(H_max),
        'per_seed': all_seed_data,
        'aggregated_correlations': agg_corrs,
    }

    save_results(results, args.output_dir, '11_exact_trA.json')


if __name__ == '__main__':
    main()
