"""
Statistical significance testing with Holm-Bonferroni correction.

Implements pairwise Wilcoxon signed-rank tests between NSGA-III and each
baseline on each metric per environment. P-values are adjusted using the
Holm-Bonferroni step-down procedure before applying the alpha = 0.05
threshold.

Reproduces the statistical annotations (asterisks) in Tables 2–6 of the
paper.

Usage
-----
python analysis/statistical_tests.py --results-dir results/
python analysis/statistical_tests.py --results-dir results/ --metric hypervolume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.pareto_metrics import compute_all_metrics


def load_seed_metrics(
    results_dir: Path,
    env_prefix: str,
    algorithm: str,
    n_seeds: int = 10,
    true_front: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """Collect per-seed metric values for one (env, algorithm) pair."""
    hvs, spars, covs = [], [], []
    for seed in range(n_seeds):
        tag = f"{algorithm}_seed{seed:02d}"
        sub_dir = results_dir / f"{env_prefix}_{algorithm}_seed{seed:02d}"
        json_path = sub_dir / f"{tag}_pareto_front.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        pts = np.array([[r["performance"], r["interpretability"]] for r in data])
        if len(pts) == 0:
            continue
        m = compute_all_metrics(pts, true_front=true_front)
        hvs.append(m["hypervolume"])
        spars.append(m["sparsity"])
        if not np.isnan(m.get("coverage", float("nan"))):
            covs.append(m["coverage"])
    return {"hypervolume": hvs, "sparsity": spars, "coverage": covs}


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Holm-Bonferroni step-down multiple testing correction.

    Parameters
    ----------
    p_values : List of raw p-values.
    alpha    : Family-wise error rate.

    Returns
    -------
    List of booleans; True means the null hypothesis is rejected (significant).
    """
    k = len(p_values)
    if k == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    rejected = np.zeros(k, dtype=bool)
    for i, p in enumerate(sorted_p):
        threshold = alpha / (k - i)
        if p <= threshold:
            rejected[order[i]] = True
        else:
            break  # stop as soon as a null is not rejected
    return rejected.tolist()


def wilcoxon_test(x: List[float], y: List[float]) -> float:
    """
    Two-sided Wilcoxon signed-rank test.

    Returns the p-value, or 1.0 if the test cannot be performed
    (e.g., fewer than 6 paired samples).
    """
    try:
        from scipy import stats
        if len(x) < 6 or len(y) < 6 or len(x) != len(y):
            return 1.0
        _, p = stats.wilcoxon(x, y, alternative="two-sided")
        return float(p)
    except Exception:
        return 1.0


ENVIRONMENTS = [
    "cartpole_v1",
    "deep_sea_treasure_v0",
    "hopper_v3",
    "mo_halfcheetah_v4",
]

BASELINES = ["moead", "weighted_sum"]
METRICS = ["hypervolume", "sparsity"]


def run_tests(
    results_dir: Path,
    n_seeds: int = 10,
    alpha: float = 0.05,
) -> None:
    """Run all pairwise tests and print a formatted results table."""
    results_dir = Path(results_dir)

    print("=" * 70)
    print(f"Statistical Tests: Wilcoxon signed-rank + Holm-Bonferroni (α={alpha})")
    print("=" * 70)

    for env in ENVIRONMENTS:
        print(f"\n{'─' * 60}")
        print(f"Environment: {env}")
        print(f"{'─' * 60}")

        nsga3_metrics = load_seed_metrics(results_dir, env, "nsga3", n_seeds)

        # Collect all p-values for Holm-Bonferroni correction
        p_values: List[float] = []
        test_records: List[Dict] = []

        for baseline in BASELINES:
            baseline_metrics = load_seed_metrics(results_dir, env, baseline, n_seeds)
            for metric in METRICS:
                x = nsga3_metrics.get(metric, [])
                y = baseline_metrics.get(metric, [])
                # Align lengths
                n = min(len(x), len(y))
                if n > 0:
                    p = wilcoxon_test(x[:n], y[:n])
                else:
                    p = 1.0
                p_values.append(p)
                test_records.append(
                    {
                        "baseline": baseline,
                        "metric": metric,
                        "nsga3_mean": float(np.mean(x)) if x else float("nan"),
                        "baseline_mean": float(np.mean(y)) if y else float("nan"),
                        "p_value": p,
                        "n": n,
                    }
                )

        rejected = holm_bonferroni(p_values, alpha=alpha)

        # Print results
        header = f"{'Baseline':<16} {'Metric':<14} {'NSGA-III':>10} {'Baseline':>10} {'p-value':>10} {'Sig.':>6}"
        print(header)
        print("-" * len(header))
        for rec, sig in zip(test_records, rejected):
            star = " *" if sig else "  "
            n3 = f"{rec['nsga3_mean']:.4f}" if not np.isnan(rec['nsga3_mean']) else "  N/A"
            bl = f"{rec['baseline_mean']:.4f}" if not np.isnan(rec['baseline_mean']) else "  N/A"
            pv = f"{rec['p_value']:.4f}"
            print(
                f"{rec['baseline']:<16} {rec['metric']:<14} {n3:>10} {bl:>10} {pv:>10}{star}"
            )

    print("\n* = significant after Holm-Bonferroni correction at α=" + str(alpha))


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_tests(args.results_dir, n_seeds=args.n_seeds, alpha=args.alpha)


if __name__ == "__main__":
    main()
