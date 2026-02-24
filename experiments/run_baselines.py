"""
Baseline comparison script.

Runs weighted sum scalarization and VIPER (where applicable) for a given
environment configuration, then compares against saved NSGA-III / MOEA/D
results.

Usage
-----
python experiments/run_baselines.py --config configs/cartpole.yaml --seed 42
python experiments/run_baselines.py --config configs/dst.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.morl_framework import MORLFramework
from src.baselines.viper import VIPERBaseline
from src.baselines.weighted_sum import WeightedSumBaseline
from src.evaluation.pareto_metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baselines")


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_weighted_sum(framework: MORLFramework, output_dir: Path, seed: int) -> None:
    """Run weighted sum scalarization baseline."""
    logger.info("Running Weighted Sum baseline...")
    alphas = framework.config.get("baselines", {}).get(
        "weighted_sum_alphas", [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    baseline = WeightedSumBaseline(framework, alphas=alphas)
    results = baseline.run()

    # Collect objectives for Pareto front quality metrics
    front_obj = np.array([[r["performance"], r["interpretability"]] for r in results])
    true_front = framework.env_spec.true_pareto_front
    metrics = compute_all_metrics(front_obj, true_front=true_front)

    logger.info("Weighted Sum | HV=%.4f | S=%.4f", metrics["hypervolume"], metrics["sparsity"])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "alpha": r["alpha"],
            "performance": r["performance"],
            "interpretability": r["interpretability"],
            "tree_summary": r.get("tree_summary"),
        }
        for r in results
    ]
    with open(output_dir / f"weighted_sum_seed{seed:02d}.json", "w") as f:
        json.dump({"results": summary, "metrics": metrics}, f, indent=2)


def run_viper(framework: MORLFramework, output_dir: Path, seed: int) -> None:
    """Run VIPER baseline (single-objective environments only)."""
    if framework.env_spec.is_multi_objective:
        logger.info("Skipping VIPER (not applicable to multi-objective environments).")
        return

    logger.info("Running VIPER baseline...")
    viper_cfg = framework.config.get("baselines", {}).get("viper", {})
    baseline = VIPERBaseline(
        framework=framework,
        max_depth=viper_cfg.get("max_depth", 5),
        n_ppo_steps=viper_cfg.get("n_ppo_steps", 100_000),
        n_dagger_iters=viper_cfg.get("n_dagger_iters", 10),
        n_dagger_episodes=viper_cfg.get("n_dagger_episodes", 20),
    )

    try:
        result = baseline.run()
        logger.info(
            "VIPER | neural_perf=%.3f | tree_perf=%.3f | interp=%.3f",
            result["neural_performance"],
            result["tree_performance"],
            result["interpretability"],
        )
        with open(output_dir / f"viper_seed{seed:02d}.json", "w") as f:
            json.dump(result, f, indent=2)
    except ImportError as e:
        logger.warning("VIPER skipped: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/baselines/<env>/)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["weighted_sum", "viper"],
        choices=["weighted_sum", "viper"],
        help="Which baselines to run",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = load_config(args.config)
    env_id = config["environment"]["id"].replace("/", "_").replace("-", "_").lower()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / "baselines" / env_id

    framework = MORLFramework(config)

    if "weighted_sum" in args.baselines:
        run_weighted_sum(framework, output_dir, args.seed)

    if "viper" in args.baselines:
        run_viper(framework, output_dir, args.seed)


if __name__ == "__main__":
    main()
