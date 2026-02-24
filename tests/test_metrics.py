"""
Unit tests for interpretability metrics and Pareto front quality metrics.

Run with:
    pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.pareto_metrics import (
    hypervolume,
    non_dominated_sort,
    pareto_coverage,
    sparsity,
)
from src.metrics.interpretability import (
    compute_causal_sufficiency,
    compute_composite_interpretability,
    compute_depth_metric,
    compute_rules_metric,
    compute_temporal_coherence,
    renormalize_weights,
)
from src.policies.decision_tree import DecisionNode, DecisionTreePolicy, LeafNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_simple_tree():
    """
    Build a small decision tree:
        if s[0] <= 0.5:
            return 0
        else:
            return 1
    Depth = 1, rules = 2, features = {0}
    """
    leaf_left = LeafNode(action=0)
    leaf_right = LeafNode(action=1)
    root = DecisionNode(feature_idx=0, threshold=0.5)
    root.left = leaf_left
    root.right = leaf_right
    return root


def make_deep_tree():
    """
    Build a depth-2 tree with 4 leaves.
    """
    l1 = LeafNode(0)
    l2 = LeafNode(1)
    l3 = LeafNode(2)
    l4 = LeafNode(3)

    inner_left = DecisionNode(0, 0.25)
    inner_left.left = l1
    inner_left.right = l2

    inner_right = DecisionNode(1, 0.75)
    inner_right.left = l3
    inner_right.right = l4

    root = DecisionNode(0, 0.5)
    root.left = inner_left
    root.right = inner_right
    return root


class FakeActionSpace:
    n = 2


def wrap_tree(root, state_dim=4):
    return DecisionTreePolicy(root, state_dim=state_dim, action_space=FakeActionSpace())


# ---------------------------------------------------------------------------
# Depth metric
# ---------------------------------------------------------------------------


class TestDepthMetric:
    def test_single_node_tree(self):
        tree = wrap_tree(LeafNode(0))
        score = compute_depth_metric(tree, d_min=1.0, d_max=5.0)
        # depth = 0; score = 1 - (0 - 1) / (5 - 1) = 1 + 0.25 = 1.25 → clipped to 1.0
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_depth_one_tree(self):
        tree = wrap_tree(make_simple_tree())
        score = compute_depth_metric(tree, d_min=1.0, d_max=5.0)
        # depth = 1; score = 1 - (1 - 1) / (5 - 1) = 1.0
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_max_depth_tree(self):
        tree = wrap_tree(make_simple_tree())
        # Set d_max = 1 (same as depth) → score = 0.0
        score = compute_depth_metric(tree, d_min=0.0, d_max=1.0)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_output_in_unit_interval(self):
        root = make_deep_tree()
        tree = wrap_tree(root)
        score = compute_depth_metric(tree, d_min=1.0, d_max=10.0)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Rules metric
# ---------------------------------------------------------------------------


class TestRulesMetric:
    def test_two_rules(self):
        tree = wrap_tree(make_simple_tree())
        # 2 leaves = r_min → score = 1.0
        score = compute_rules_metric(tree, r_min=2.0, r_max=16.0)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_four_rules(self):
        tree = wrap_tree(make_deep_tree())
        # 4 leaves; score = 1 - (4 - 2) / (16 - 2) = 1 - 2/14 ≈ 0.857
        score = compute_rules_metric(tree, r_min=2.0, r_max=16.0)
        expected = 1.0 - (4 - 2) / (16 - 2)
        assert score == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Temporal coherence
# ---------------------------------------------------------------------------


class TestTemporalCoherence:
    def test_constant_actions_discrete(self):
        tree = wrap_tree(LeafNode(0))
        # All actions the same → coherence = 1.0
        trajs = [[0, 0, 0, 0], [0, 0, 0]]
        score = compute_temporal_coherence(tree, trajs, is_continuous=False)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_alternating_actions_discrete(self):
        tree = wrap_tree(LeafNode(0))
        # Alternating → every consecutive pair differs → coherence = 0.0
        trajs = [[0, 1, 0, 1]]
        score = compute_temporal_coherence(tree, trajs, is_continuous=False)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_continuous_identical_actions(self):
        tree = wrap_tree(LeafNode(np.array([0.5, 0.5])))
        a = np.array([0.5, 0.5])
        trajs = [[a, a, a]]
        score = compute_temporal_coherence(
            tree, trajs, is_continuous=True, action_dim=2, a_max=1.0
        )
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_empty_trajectories(self):
        tree = wrap_tree(LeafNode(0))
        score = compute_temporal_coherence(tree, [], is_continuous=False)
        assert score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Causal sufficiency
# ---------------------------------------------------------------------------


class TestCausalSufficiency:
    def test_full_coverage(self):
        # 2-node tree uses both features of a 2D state space
        root = DecisionNode(0, 0.5)
        root.left = DecisionNode(1, 0.3)
        root.left.left = LeafNode(0)
        root.left.right = LeafNode(1)
        root.right = LeafNode(2)
        tree = wrap_tree(root, state_dim=2)
        score = compute_causal_sufficiency(tree, state_dim=2)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_partial_coverage(self):
        # Tree uses 1 of 4 features
        tree = wrap_tree(make_simple_tree(), state_dim=4)
        score = compute_causal_sufficiency(tree, state_dim=4)
        assert score == pytest.approx(0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# Composite metric
# ---------------------------------------------------------------------------


class TestCompositeMetric:
    def test_output_in_unit_interval(self):
        tree = wrap_tree(make_simple_tree(), state_dim=4)
        trajs = [[0, 0, 0], [1, 1]]
        score, components = compute_composite_interpretability(
            tree, trajs, state_dim=4
        )
        assert 0.0 <= score <= 1.0
        for k in ("M_depth", "M_rules", "M_temporal", "M_causal", "composite"):
            assert k in components
            assert 0.0 <= components[k] <= 1.0

    def test_weights_sum_to_one(self):
        # Default weights (0.3, 0.3, 0.2, 0.2) must sum to 1
        from src.metrics.interpretability import compute_composite_interpretability
        w = (0.3, 0.3, 0.2, 0.2)
        assert sum(w) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Weight renormalization (ablation)
# ---------------------------------------------------------------------------


class TestRenormalization:
    def test_ablate_temporal(self):
        w = renormalize_weights((0.3, 0.3, 0.2, 0.2), ablated_indices=[2])
        assert w[2] == pytest.approx(0.0, abs=1e-9)
        assert sum(w) == pytest.approx(1.0, abs=1e-6)

    def test_ablate_structural(self):
        w = renormalize_weights((0.3, 0.3, 0.2, 0.2), ablated_indices=[0, 1])
        assert w[0] == pytest.approx(0.0, abs=1e-9)
        assert w[1] == pytest.approx(0.0, abs=1e-9)
        assert sum(w) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Pareto front metrics
# ---------------------------------------------------------------------------


class TestHypervolume:
    def test_single_point(self):
        front = np.array([[2.0, 0.8]])
        ref = np.array([0.0, 0.0])
        hv = hypervolume(front, ref)
        assert hv == pytest.approx(2.0 * 0.8, abs=1e-9)

    def test_two_points(self):
        # Two non-dominated points forming an L-shaped front
        front = np.array([[3.0, 0.5], [1.0, 0.9]])
        ref = np.array([0.0, 0.0])
        hv = hypervolume(front, ref)
        # Sorted by f1 desc: [3.0, 0.5], [1.0, 0.9]
        # Rectangle 1: (3.0 - 0) * (0.5 - 0.0) = 1.5
        # Rectangle 2: (1.0 - 0) * (0.9 - 0.5) = 0.4
        assert hv == pytest.approx(1.9, abs=1e-9)

    def test_empty_front(self):
        assert hypervolume(np.empty((0, 2)), np.array([0.0, 0.0])) == 0.0


class TestNonDominatedSort:
    def test_all_dominated(self):
        # [1,1] is dominated by [2,2]
        objs = np.array([[1.0, 1.0], [2.0, 2.0]])
        nd = non_dominated_sort(objs)
        assert list(nd) == [1]

    def test_none_dominated(self):
        # [3,0] and [0,3] are mutually non-dominating
        objs = np.array([[3.0, 0.0], [0.0, 3.0]])
        nd = non_dominated_sort(objs)
        assert set(nd) == {0, 1}


class TestCoverage:
    def test_perfect_coverage(self):
        true_front = np.array([[1.0, 1.0], [2.0, 0.5]])
        approx = np.array([[1.5, 1.5], [2.5, 1.0]])  # dominates all true front points
        cov = pareto_coverage(approx, true_front)
        assert cov == pytest.approx(1.0, abs=1e-6)

    def test_zero_coverage(self):
        true_front = np.array([[5.0, 5.0]])
        approx = np.array([[1.0, 1.0]])  # does not dominate
        cov = pareto_coverage(approx, true_front)
        assert cov == pytest.approx(0.0, abs=1e-6)


class TestSparsity:
    def test_single_point(self):
        assert sparsity(np.array([[1.0, 0.5]])) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_two_points(self):
        s = sparsity(np.array([[0.0, 0.0], [1.0, 1.0]]))
        # Normalized distance = sqrt(2), both points have NN distance sqrt(2)
        # sparsity = sqrt(2)
        assert s > 0.0
