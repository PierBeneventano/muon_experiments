"""Utility functions for experiment scripts."""
import argparse
import json
import os
import numpy as np


def set_seed(seed):
    """Set numpy random seed for reproducibility."""
    np.random.seed(seed)


def save_results(results, output_dir, filename):
    """
    Save results dict to JSON file with indent=2.
    Creates output_dir if it does not exist.
    Converts numpy types to native Python types for JSON serialization.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"Results saved to {path}")
    return path


def load_results(path):
    """Load results from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_parser(description):
    """
    Return an argparse.ArgumentParser with common experiment arguments.
    --seed: base random seed (default 42)
    --output_dir: directory to write results (default './results')
    --n_seeds: number of seeds to run (default 10)
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results (default: ./results)')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Number of seeds to run (default: 10)')
    return parser


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
