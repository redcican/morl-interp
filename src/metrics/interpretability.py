"""
Composite interpretability proxy metric and its four components.

The composite score is defined as:

    I(T) = w1 * M_depth + w2 * M_rules + w3 * M_temporal + w4 * M_causal

where each component is in [0, 1] and the weights sum to 1.

Components
----------
M_depth   : Normalized inverse tree depth (shallower = more interpretable).
M_rules   : Normalized inverse leaf count (fewer rules = more interpretable).
M_temporal: Action consistency along trajectories.
             For discrete action spaces: fraction of consecutive steps with
             the same action, averaged over trajectories.
             For continuous action spaces: normalized action-change magnitude
             (Equation 15 in the paper; see continuous action adaptation note).
M_causal  : Fraction of state dimensions tested by internal nodes, measuring
             how completely the policy accounts for the state.

References
----------
Definitions 4–7 and Section 4.2 of the paper; the continuous-domain
adaptation is the revised Equation 15 added in response to Reviewer 3.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..policies.decision_tree import DecisionTreePolicy


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_depth_metric(
    tree: DecisionTreePolicy,
    d_min: float = 1.0,
    d_max: float = 10.0,
) -> float:
    """
    M_depth = 1 - (d(T) - d_min) / (d_max - d_min)

    A tree at minimum depth scores 1; at maximum depth scores 0.
    Values outside [d_min, d_max] are clipped.
    """
    if d_max <= d_min:
        return 1.0
    d = float(tree.get_depth())
    score = 1.0 - (d - d_min) / (d_max - d_min)
    return float(np.clip(score, 0.0, 1.0))


def compute_rules_metric(
    tree: DecisionTreePolicy,
    r_min: float = 2.0,
    r_max: float = 32.0,
) -> float:
    """
    M_rules = 1 - (|L(T)| - r_min) / (r_max - r_min)

    A tree with r_min leaves scores 1; with r_max leaves scores 0.
    """
    if r_max <= r_min:
        return 1.0
    r = float(tree.get_num_rules())
    score = 1.0 - (r - r_min) / (r_max - r_min)
    return float(np.clip(score, 0.0, 1.0))


def compute_temporal_coherence(
    tree: DecisionTreePolicy,
    trajectories: List[List],
    is_continuous: bool = False,
    action_dim: int = 1,
    a_max: float = 1.0,
) -> float:
    """
    M_temporal: average action consistency over a set of trajectories.

    Discrete action spaces (Definition 4.6 in the paper):
        indicator = 1 if a_t == a_{t+1}, else 0

    Continuous action spaces (Equation 15, continuous adaptation):
        indicator = max(0, 1 - ||a_t - a_{t+1}||_2 / (sqrt(|A|) * a_max))

    The metric is averaged first over consecutive pairs within each
    trajectory, then over trajectories.
    """
    if not trajectories:
        return 0.0

    coherence_values: List[float] = []
    norm_factor = float(np.sqrt(action_dim)) * a_max

    for traj in trajectories:
        if len(traj) < 2:
            continue
        step_scores: List[float] = []
        for t in range(len(traj) - 1):
            a_t = traj[t]
            a_t1 = traj[t + 1]
            if is_continuous:
                a_t = np.asarray(a_t, dtype=float)
                a_t1 = np.asarray(a_t1, dtype=float)
                diff = float(np.linalg.norm(a_t - a_t1))
                indicator = max(0.0, 1.0 - diff / (norm_factor + 1e-8))
            else:
                indicator = 1.0 if a_t == a_t1 else 0.0
            step_scores.append(indicator)
        if step_scores:
            coherence_values.append(float(np.mean(step_scores)))

    return float(np.mean(coherence_values)) if coherence_values else 0.0


def compute_causal_sufficiency(
    tree: DecisionTreePolicy,
    state_dim: int,
) -> float:
    """
    M_causal = |F(T)| / |S|

    where F(T) is the set of state feature indices tested by internal nodes.

    This metric is designed for environments with a moderate number of state
    dimensions (e.g., CartPole: 4D, Hopper: 11D) where all features are
    plausibly task-relevant. In high-dimensional domains the denominator
    should be replaced by the count of pre-identified causally relevant
    features rather than the raw state dimension.

    Scope note (added in response to Reviewer 3, Question 3):
    In high-dimensional or image-based domains, maximizing this metric would
    push policies to use many features, which is contrary to interpretability.
    The metric is intentionally scoped to moderate-dimensional environments.
    """
    if state_dim == 0:
        return 1.0
    features = tree.get_features_used()
    return float(len(features)) / float(state_dim)


# ---------------------------------------------------------------------------
# Composite metric
# ---------------------------------------------------------------------------


def compute_composite_interpretability(
    tree: DecisionTreePolicy,
    trajectories: List[List],
    state_dim: int,
    weights: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2),
    d_min: float = 1.0,
    d_max: float = 10.0,
    r_min: float = 2.0,
    r_max: float = 32.0,
    is_continuous: bool = False,
    action_dim: int = 1,
    a_max: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the composite interpretability proxy metric I(T).

    I(T) = w1 * M_depth + w2 * M_rules + w3 * M_temporal + w4 * M_causal

    Default weights (w1, w2, w3, w4) = (0.3, 0.3, 0.2, 0.2) assign equal
    importance to the two structural components and equal importance to the
    two behavioral components, with structural properties weighted slightly
    higher to reflect their stronger association with simulatability.

    Parameters
    ----------
    tree        : The decision tree policy to evaluate.
    trajectories: List of action sequences [a_0, a_1, ...] from evaluation
                  rollouts, used to compute M_temporal.
    state_dim   : Observation space dimension, used to compute M_causal.
    weights     : (w1, w2, w3, w4) with sum == 1.
    d_min, d_max: Depth range for normalization.
    r_min, r_max: Rule count range for normalization.
    is_continuous: True for continuous action spaces (Hopper, MO-HalfCheetah).
    action_dim  : Action space dimensionality (for continuous normalization).
    a_max       : Per-dimension action bound (for continuous normalization).

    Returns
    -------
    composite : Scalar composite score in [0, 1].
    components: Dictionary with individual metric values and composite score.
    """
    w1, w2, w3, w4 = weights

    m_depth = compute_depth_metric(tree, d_min, d_max)
    m_rules = compute_rules_metric(tree, r_min, r_max)
    m_temporal = compute_temporal_coherence(
        tree, trajectories, is_continuous, action_dim, a_max
    )
    m_causal = compute_causal_sufficiency(tree, state_dim)

    composite = w1 * m_depth + w2 * m_rules + w3 * m_temporal + w4 * m_causal
    composite = float(np.clip(composite, 0.0, 1.0))

    components = {
        "M_depth": m_depth,
        "M_rules": m_rules,
        "M_temporal": m_temporal,
        "M_causal": m_causal,
        "composite": composite,
    }
    return composite, components


# ---------------------------------------------------------------------------
# Ablation: proportional weight renormalization
# ---------------------------------------------------------------------------


def renormalize_weights(
    weights: Tuple[float, float, float, float],
    ablated_indices: List[int],
) -> Tuple[float, float, float, float]:
    """
    Set the weights at ablated_indices to zero, then renormalize the
    remaining weights proportionally so they sum to 1.

    This implements the proportional renormalization described in the ablation
    study: freed weight mass is distributed in proportion to the remaining
    weights, preserving their relative importance.

    Parameters
    ----------
    weights        : Original weight tuple (w1, w2, w3, w4).
    ablated_indices: Indices (0-based) of components to ablate.

    Returns
    -------
    Renormalized weight tuple with ablated weights set to 0.
    """
    w = list(weights)
    for i in ablated_indices:
        w[i] = 0.0
    total = sum(w)
    if total > 0:
        w = [wi / total for wi in w]
    return tuple(w)
