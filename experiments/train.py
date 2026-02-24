"""
Main training script for multi-objective evolutionary interpretable RL.

Usage
-----
# Train with NSGA-III on CartPole, seed 42
python experiments/train.py --config configs/cartpole.yaml --algorithm nsga3 --seed 42

# Train with MOEA/D on Deep Sea Treasure
python experiments/train.py --config configs/dst.yaml --algorithm moead --seed 0

# Train with both algorithms (runs sequentially)
python experiments/train.py --config configs/hopper.yaml --algorithm both --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.moead import MOEADOptimizer
from src.algorithms.morl_framework import MORLFramework
from src.algorithms.nsga3 import NSGA3Optimizer
from src.evaluation.pareto_metrics import compute_all_metrics, non_dominated_sort

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_results(results: List[Dict], output_dir: Path, tag: str) -> None:
    """Save Pareto front results to JSON and pickle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON-serializable summary
    summary = []
    for r in results:
        s = {
            "performance": r.get("performance"),
            "interpretability": r.get("interpretability"),
            "objectives": r.get("objectives"),
        }
        if "tree_summary" in r:
            s["tree_summary"] = r["tree_summary"]
        if "components" in r:
            s["components"] = r["components"]
        if "alpha" in r:
            s["alpha"] = r["alpha"]
        summary.append(s)

    json_path = output_dir / f"{tag}_pareto_front.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results (including policies) as pickle
    pkl_path = output_dir / f"{tag}_results.pkl"
    with open(pkl_path, "wb") as f:
        # Exclude non-serializable action_space from policies
        save_data = []
        for r in results:
            rd = {k: v for k, v in r.items() if k != "policy"}
            save_data.append(rd)
        pickle.dump(save_data, f)

    logger.info("Results saved to %s", output_dir)


def run_experiment(
    config: Dict,
    algorithm: str,
    seed: int,
    output_dir: Path,
) -> None:
    """Run a single experiment with the given algorithm and seed."""
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("Environment : %s", config["environment"]["id"])
    logger.info("Algorithm   : %s", algorithm.upper())
    logger.info("Seed        : %d", seed)
    logger.info("=" * 60)

    # Build framework
    framework = MORLFramework(config)

    t_start = time.time()

    if algorithm == "nsga3":
        optimizer = NSGA3Optimizer(framework)
    elif algorithm == "moead":
        optimizer = MOEADOptimizer(framework)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    results = optimizer.run()
    elapsed = time.time() - t_start

    # Compute final metrics
    if results:
        front_obj = np.array([[r["performance"], r["interpretability"]] for r in results])
        true_front = framework.env_spec.true_pareto_front
        metrics = compute_all_metrics(front_obj, true_front=true_front)
    else:
        metrics = {"hypervolume": 0.0, "sparsity": 0.0, "coverage": float("nan")}

    logger.info("--- Final Metrics ---")
    logger.info("  Pareto front size : %d", len(results))
    logger.info("  Hypervolume       : %.4f", metrics["hypervolume"])
    logger.info("  Sparsity          : %.4f", metrics["sparsity"])
    if not np.isnan(metrics.get("coverage", float("nan"))):
        logger.info("  Coverage          : %.4f", metrics["coverage"])
    logger.info("  Wall time         : %.1f s", elapsed)

    # Save results
    tag = f"{algorithm}_seed{seed:02d}"
    save_results(results, output_dir, tag)

    # Save training history
    hist_path = output_dir / f"{tag}_history.json"
    with open(hist_path, "w") as f:
        json.dump(framework.history, f, indent=2)

    # Save final summary metrics
    summary = {
        "algorithm": algorithm,
        "seed": seed,
        "env_id": config["environment"]["id"],
        "pareto_size": len(results),
        "elapsed_seconds": elapsed,
        **metrics,
    }
    summary_path = output_dir / f"{tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multi-objective evolutionary interpretable RL"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--algorithm",
        default="nsga3",
        choices=["nsga3", "moead", "both"],
        help="Evolutionary algorithm to use (default: nsga3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results (default: results/<env>_<alg>_seed<n>/)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    env_id = config["environment"]["id"].replace("/", "_").replace("-", "_").lower()

    algorithms = ["nsga3", "moead"] if args.algorithm == "both" else [args.algorithm]

    for alg in algorithms:
        if args.output_dir is not None:
            out_dir = Path(args.output_dir)
        else:
            out_dir = Path("results") / f"{env_id}_{alg}_seed{args.seed:02d}"

        run_experiment(config, alg, args.seed, out_dir)


if __name__ == "__main__":
    main()
