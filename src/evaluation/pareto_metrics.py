"""
Pareto front quality metrics for multi-objective evaluation.

Metrics
-------
Hypervolume (HV)
    Volume of the objective space dominated by the approximated Pareto front
    relative to a reference point. Higher is better.
    Two-dimensional closed-form implementation.

Sparsity (S)
    Average nearest-neighbor distance among Pareto front points in normalized
    objective space. Higher sparsity = more spread-out front.

Coverage (Cov)
    Fraction of the *true* Pareto front that is weakly dominated by at least
    one solution in the approximated front. Only applicable when the true
    front is known (i.e., Deep Sea Treasure in our study).

Non-dominated sort
    Standard filter that returns indices of Pareto-optimal solutions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hypervolume (2D)
# ---------------------------------------------------------------------------


def hypervolume(
    pareto_front: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """
    Compute the 2-D hypervolume indicator.

    Parameters
    ----------
    pareto_front : Array of shape (n, 2). Each row is an objective vector
                   [perf, interp], where higher values are better.
    ref_point    : Reference point that is weakly dominated by all solutions.
                   Typically set to (min - epsilon) across objectives.

    Returns
    -------
    Hypervolume as a non-negative float.
    """
    if len(pareto_front) == 0:
        return 0.0

    # Keep only solutions that dominate the reference point
    mask = np.all(pareto_front > ref_point, axis=1)
    front = pareto_front[mask]
    if len(front) == 0:
        return 0.0

    # Sort by first objective descending, then sweep to sum rectangles
    order = np.argsort(-front[:, 0])
    front = front[order]

    hv = 0.0
    prev_y = ref_point[1]
    for point in front:
        if point[1] > prev_y:
            hv += (point[0] - ref_point[0]) * (point[1] - prev_y)
            prev_y = point[1]

    return float(hv)


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------


def sparsity(pareto_front: np.ndarray) -> float:
    """
    Average nearest-neighbor distance among Pareto front solutions,
    computed in normalized objective space.

    A higher value indicates a more spread-out front.
    Returns 0 for single-point fronts.
    """
    if len(pareto_front) < 2:
        return 0.0

    # Normalize to [0, 1] per objective
    f_min = pareto_front.min(axis=0)
    f_max = pareto_front.max(axis=0)
    denom = f_max - f_min
    denom[denom == 0.0] = 1.0
    normalized = (pareto_front - f_min) / denom

    total = 0.0
    n = len(normalized)
    for i in range(n):
        dists = np.linalg.norm(normalized - normalized[i], axis=1)
        dists[i] = np.inf
        total += dists.min()

    return float(total / n)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def pareto_coverage(
    approx_front: np.ndarray,
    true_front: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Fraction of the true Pareto front weakly dominated by the approximated
    front.

    A true-front point p is covered if there exists an approx-front point q
    such that q >= p (element-wise, within epsilon tolerance).

    Note: the Coverage metric requires a known reference true Pareto front.
    In this study it is only reported for Deep Sea Treasure, which has an
    analytically established front.

    Parameters
    ----------
    approx_front : Array of shape (m, 2) — the approximated Pareto front.
    true_front   : Array of shape (k, 2) — the ground-truth Pareto front.
    epsilon      : Numerical tolerance for dominance comparison.

    Returns
    -------
    Coverage in [0, 1].
    """
    if len(true_front) == 0:
        return 1.0
    if len(approx_front) == 0:
        return 0.0

    covered = 0
    for tp in true_front:
        for ap in approx_front:
            if np.all(ap >= tp - epsilon):
                covered += 1
                break

    return float(covered) / float(len(true_front))


# ---------------------------------------------------------------------------
# Non-dominated sort
# ---------------------------------------------------------------------------


def non_dominated_sort(objectives: np.ndarray) -> np.ndarray:
    """
    Return indices of non-dominated (Pareto-optimal) solutions.

    A solution i is non-dominated if no other solution j satisfies
    objectives[j] >= objectives[i] (element-wise) with strict inequality
    in at least one objective.

    Parameters
    ----------
    objectives : Array of shape (n, m) with m objectives (higher is better).

    Returns
    -------
    Array of indices of non-dominated solutions.
    """
    n = len(objectives)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(objectives[j] >= objectives[i]) and np.any(
                objectives[j] > objectives[i]
            ):
                dominated[i] = True
                break

    return np.where(~dominated)[0]


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def compute_all_metrics(
    approx_front: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    true_front: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all Pareto front quality metrics in one call.

    Parameters
    ----------
    approx_front : Array of shape (n, 2) — the non-dominated solution set.
    ref_point    : Reference point for hypervolume. Defaults to
                   min(approx_front) - 0.1 if not provided.
    true_front   : Ground-truth Pareto front for Coverage computation.
                   Pass None if not available (returns NaN for Coverage).

    Returns
    -------
    Dictionary with keys: 'hypervolume', 'sparsity', 'coverage', 'front_size'.
    """
    results: Dict[str, float] = {"front_size": float(len(approx_front))}

    if len(approx_front) == 0:
        results["hypervolume"] = 0.0
        results["sparsity"] = 0.0
        results["coverage"] = float("nan")
        return results

    if ref_point is None:
        ref_point = approx_front.min(axis=0) - 0.1

    results["hypervolume"] = hypervolume(approx_front, ref_point)
    results["sparsity"] = sparsity(approx_front)
    results["coverage"] = (
        pareto_coverage(approx_front, true_front)
        if true_front is not None
        else float("nan")
    )

    return results
