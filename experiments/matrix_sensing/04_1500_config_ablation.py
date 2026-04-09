#!/usr/bin/env python3
"""Large ablation sweep (1500-configuration).

m in {20, 50}, p in {50, 100, 200, 300, 500}, init_scale in {0.01, 0.1, 1.0}, rank in {2, 5}.
10 seeds each. 60 conditions x 10 seeds = 600 runs per optimizer (Muon vs GD).
Report fraction of conditions where Muon H > GD H.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
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
    parser = get_parser("1500-configuration ablation sweep")
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    args.n_seeds = 10

    ms = [20, 50]
    ps = [50, 100, 200, 300, 500]
    init_scales = [0.01, 0.1, 1.0]
    ranks = [2, 5]

    configs = []
    for m_val in ms:
        for p_val in ps:
            for init_scale in init_scales:
                for rank in ranks:
                    configs.append({
                        'm': m_val, 'n': m_val, 'p': p_val,
                        'init_scale': init_scale, 'rank': rank,
                    })

    print(f"Total configurations: {len(configs)}")
    print(f"Total runs per optimizer: {len(configs) * args.n_seeds}")

    muon_wins = 0
    total_configs = 0
    per_config_results = []

    for cfg_idx, cfg in enumerate(configs):
        m_val = cfg['m']
        n_val = cfg['n']
        p_val = cfg['p']
        init_scale = cfg['init_scale']
        rank = cfg['rank']
        H_max = max_entropy(m_val, n_val)

        H_muon_seeds = []
        H_gd_seeds = []

        for s_idx in range(args.n_seeds):
            seed = args.seed + s_idx
            set_seed(seed)
            W_star, A, b = generate_problem(m_val, n_val, p_val, rank, seed=seed)

            for opt_name in ['muon', 'gd']:
                set_seed(seed + 10000)
                W_init = np.random.randn(m_val, n_val) * init_scale
                W_final = run_optimizer(opt_name, W_init, A, b, args.n_steps, args.lr)

                if np.any(np.isnan(W_final)) or np.any(np.isinf(W_final)):
                    H = 0.0
                else:
                    H = spectral_entropy(W_final)

                if opt_name == 'muon':
                    H_muon_seeds.append(float(H))
                else:
                    H_gd_seeds.append(float(H))

        mean_muon = np.mean(H_muon_seeds)
        mean_gd = np.mean(H_gd_seeds)
        muon_wins_here = mean_muon > mean_gd
        if muon_wins_here:
            muon_wins += 1
        total_configs += 1

        record = {
            'config': cfg,
            'H_muon_mean': float(mean_muon),
            'H_gd_mean': float(mean_gd),
            'H_advantage': float(mean_muon - mean_gd),
            'muon_wins': bool(muon_wins_here),
            'H_muon_seeds': H_muon_seeds,
            'H_gd_seeds': H_gd_seeds,
        }
        per_config_results.append(record)

        if (cfg_idx + 1) % 10 == 0 or cfg_idx == 0:
            print(f"  Config {cfg_idx+1}/{len(configs)}: "
                  f"m={m_val} p={p_val} scale={init_scale} rank={rank} | "
                  f"Muon H={mean_muon:.4f}, GD H={mean_gd:.4f}, "
                  f"win={muon_wins_here}")

    win_fraction = muon_wins / total_configs
    print(f"\n=== Summary ===")
    print(f"Muon wins: {muon_wins}/{total_configs} ({win_fraction:.1%})")

    # Breakdown by factor
    breakdowns = {}
    for factor_name, factor_values, factor_key in [
        ('m', ms, 'm'), ('p', ps, 'p'),
        ('init_scale', init_scales, 'init_scale'), ('rank', ranks, 'rank')
    ]:
        breakdown = {}
        for val in factor_values:
            matching = [r for r in per_config_results if r['config'][factor_key] == val]
            wins = sum(1 for r in matching if r['muon_wins'])
            breakdown[str(val)] = {
                'n_configs': len(matching),
                'muon_wins': wins,
                'win_fraction': wins / len(matching) if matching else 0.0,
            }
        breakdowns[factor_name] = breakdown

    results = {
        'experiment': '1500_config_ablation',
        'config': {
            'ms': ms, 'ps': ps, 'init_scales': init_scales, 'ranks': ranks,
            'n_steps': args.n_steps, 'lr': args.lr, 'n_seeds': args.n_seeds,
            'n_configs': total_configs, 'base_seed': args.seed,
        },
        'overall_win_fraction': float(win_fraction),
        'muon_wins': muon_wins,
        'total_configs': total_configs,
        'breakdowns': breakdowns,
        'per_config': per_config_results,
    }

    save_results(results, args.output_dir, '04_1500_config_ablation.json')


if __name__ == '__main__':
    main()
