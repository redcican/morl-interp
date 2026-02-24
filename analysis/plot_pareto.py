"""
Pareto front visualization script.

Generates publication-quality plots comparing NSGA-III, MOEA/D, and weighted
sum baseline Pareto fronts for all four environments. Reproduces Figures 1–5
from the paper.

Usage
-----
python analysis/plot_pareto.py --results-dir results/
python analysis/plot_pareto.py --results-dir results/ --env cartpole_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_front(json_path: Path) -> Optional[np.ndarray]:
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        pts = [[r["performance"], r["interpretability"]] for r in data]
    else:
        pts = [[r["performance"], r["interpretability"]] for r in data.get("results", [])]
    return np.array(pts) if pts else None


def aggregate_fronts(
    results_dir: Path,
    env_prefix: str,
    algorithm: str,
    n_seeds: int = 10,
) -> Optional[np.ndarray]:
    """Collect all solutions across seeds and return the pooled non-dominated front."""
    all_pts = []
    for seed in range(n_seeds):
        tag = f"{algorithm}_seed{seed:02d}"
        sub_dir = results_dir / f"{env_prefix}_{algorithm}_seed{seed:02d}"
        front = load_front(sub_dir / f"{tag}_pareto_front.json")
        if front is not None:
            all_pts.extend(front.tolist())
    if not all_pts:
        return None
    return np.array(all_pts)


def plot_comparison(
    fronts: Dict[str, Optional[np.ndarray]],
    env_name: str,
    output_path: Path,
    x_label: str = "Task Performance",
    y_label: str = "Interpretability I(T)",
) -> None:
    """Plot multiple Pareto fronts on one axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    STYLE = {
        "nsga3":        {"color": "#1f77b4", "marker": "o", "label": "NSGA-III (ours)"},
        "moead":        {"color": "#ff7f0e", "marker": "s", "label": "MOEA/D (ours)"},
        "weighted_sum": {"color": "#2ca02c", "marker": "^", "label": "Weighted Sum"},
        "viper":        {"color": "#d62728", "marker": "D", "label": "VIPER"},
        "pirl":         {"color": "#9467bd", "marker": "P", "label": "PIRL"},
    }

    fig, ax = plt.subplots(figsize=(6, 5))

    for key, pts in fronts.items():
        if pts is None or len(pts) == 0:
            continue
        style = STYLE.get(key, {"color": "gray", "marker": "x", "label": key})
        # Sort by first objective for line plot
        order = np.argsort(pts[:, 0])
        pts_sorted = pts[order]
        ax.plot(
            pts_sorted[:, 0],
            pts_sorted[:, 1],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.5,
            markersize=6,
            alpha=0.85,
        )

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(env_name, fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


ENV_CONFIGS = {
    "cartpole_v1": {
        "display": "CartPole-v1",
        "x_label": "Task Performance (discounted return)",
        "y_label": "Interpretability I(T)",
    },
    "deep_sea_treasure_v0": {
        "display": "Deep Sea Treasure",
        "x_label": "Task Performance (Σ objectives)",
        "y_label": "Interpretability I(T)",
    },
    "hopper_v3": {
        "display": "Hopper-v3",
        "x_label": "Task Performance (discounted return)",
        "y_label": "Interpretability I(T)",
    },
    "mo_halfcheetah_v4": {
        "display": "MO-HalfCheetah",
        "x_label": "Task Performance (Σ objectives)",
        "y_label": "Interpretability I(T)",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pareto front comparison plots")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--env", default=None, help="Specific environment prefix to plot")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--output-dir", default="results/figures/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    envs = [args.env] if args.env else list(ENV_CONFIGS.keys())

    for env_prefix in envs:
        cfg = ENV_CONFIGS.get(env_prefix, {"display": env_prefix, "x_label": "Performance", "y_label": "Interpretability"})
        fronts = {}
        for alg in ["nsga3", "moead"]:
            fronts[alg] = aggregate_fronts(results_dir, env_prefix, alg, args.n_seeds)

        # Load weighted sum baseline (seed 0 only for illustration)
        ws_path = results_dir / "baselines" / env_prefix / "weighted_sum_seed00.json"
        ws_front = load_front(ws_path)
        if ws_front is not None:
            fronts["weighted_sum"] = ws_front

        plot_comparison(
            fronts,
            env_name=cfg["display"],
            output_path=output_dir / f"pareto_{env_prefix}.png",
            x_label=cfg["x_label"],
            y_label=cfg["y_label"],
        )


if __name__ == "__main__":
    main()
