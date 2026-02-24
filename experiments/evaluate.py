"""
Evaluation script: load saved results and compute Pareto front metrics.

Usage
-----
# Evaluate a saved results directory
python experiments/evaluate.py --results results/cartpole_v1_nsga3_seed42/

# Evaluate and display Pareto front plot
python experiments/evaluate.py --results results/dst_nsga3_seed00/ --plot

# Evaluate all seeds and compute statistics
python experiments/evaluate.py --results-dir results/ --env cartpole_v1 --algorithm nsga3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.pareto_metrics import compute_all_metrics


def load_pareto_front(results_dir: Path, tag: str) -> Optional[np.ndarray]:
    """Load Pareto front objectives from a JSON file."""
    json_path = results_dir / f"{tag}_pareto_front.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return np.array([[r["performance"], r["interpretability"]] for r in data])


def print_metrics(metrics: Dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Front size : {int(metrics.get('front_size', 0))}")
    print(f"{prefix}HV         : {metrics.get('hypervolume', 0.0):.4f}")
    print(f"{prefix}Sparsity   : {metrics.get('sparsity', 0.0):.4f}")
    cov = metrics.get("coverage", float("nan"))
    if not np.isnan(cov):
        print(f"{prefix}Coverage   : {cov:.4f}")


def evaluate_single(results_dir: Path, tag: str, true_front: Optional[np.ndarray] = None) -> Dict:
    front = load_pareto_front(results_dir, tag)
    if front is None or len(front) == 0:
        print(f"No results found at {results_dir / tag}")
        return {}

    metrics = compute_all_metrics(front, true_front=true_front)
    print_metrics(metrics, label=tag)
    return metrics


def evaluate_multiple_seeds(
    results_dir: Path,
    env_name: str,
    algorithm: str,
    n_seeds: int = 10,
    true_front: Optional[np.ndarray] = None,
) -> Dict:
    """Compute mean ± std across seeds."""
    all_hv, all_spar, all_cov = [], [], []

    for seed in range(n_seeds):
        tag = f"{algorithm}_seed{seed:02d}"
        sub_dir = results_dir / f"{env_name}_{algorithm}_seed{seed:02d}"
        front = load_pareto_front(sub_dir, tag)
        if front is None or len(front) == 0:
            continue
        m = compute_all_metrics(front, true_front=true_front)
        all_hv.append(m["hypervolume"])
        all_spar.append(m["sparsity"])
        if not np.isnan(m.get("coverage", float("nan"))):
            all_cov.append(m["coverage"])

    summary = {}
    if all_hv:
        summary["hv_mean"] = float(np.mean(all_hv))
        summary["hv_std"] = float(np.std(all_hv))
        summary["sparsity_mean"] = float(np.mean(all_spar))
        summary["sparsity_std"] = float(np.std(all_spar))
        if all_cov:
            summary["cov_mean"] = float(np.mean(all_cov))
            summary["cov_std"] = float(np.std(all_cov))
        print(f"\n{env_name} | {algorithm.upper()} | {len(all_hv)} seeds")
        print(f"  HV      : {summary['hv_mean']:.4f} ± {summary['hv_std']:.4f}")
        print(f"  Sparsity: {summary['sparsity_mean']:.4f} ± {summary['sparsity_std']:.4f}")
        if "cov_mean" in summary:
            print(f"  Coverage: {summary['cov_mean']:.4f} ± {summary['cov_std']:.4f}")

    return summary


def plot_pareto_front(results_dir: Path, tag: str, save: bool = True) -> None:
    """Quick Pareto front scatter plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    front = load_pareto_front(results_dir, tag)
    if front is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(front[:, 0], front[:, 1], c="steelblue", s=60, zorder=3)
    ax.set_xlabel("Task Performance", fontsize=12)
    ax.set_ylabel("Interpretability I(T)", fontsize=12)
    ax.set_title(f"Pareto Front — {tag}", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        plot_path = results_dir / f"{tag}_pareto_front.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved MORL-Interp results")
    parser.add_argument("--results", default=None, help="Path to a single results directory")
    parser.add_argument("--results-dir", default="results/", help="Root results directory")
    parser.add_argument("--env", default=None, help="Environment name prefix (for multi-seed eval)")
    parser.add_argument("--algorithm", default="nsga3", choices=["nsga3", "moead"])
    parser.add_argument("--tag", default=None, help="Results file tag (e.g. nsga3_seed00)")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--plot", action="store_true", help="Generate Pareto front plot")
    args = parser.parse_args()

    if args.results is not None:
        results_dir = Path(args.results)
        tag = args.tag or f"{args.algorithm}_seed00"
        evaluate_single(results_dir, tag)
        if args.plot:
            plot_pareto_front(results_dir, tag)

    elif args.env is not None:
        results_dir = Path(args.results_dir)
        evaluate_multiple_seeds(
            results_dir,
            args.env,
            args.algorithm,
            n_seeds=args.n_seeds,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
