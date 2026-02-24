"""
Unit tests for decision tree policy and Grammatical Evolution encoding.

Run with:
    pytest tests/test_decision_tree.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policies.decision_tree import DecisionNode, DecisionTreePolicy, LeafNode
from src.policies.ge_encoding import GrammaticalEvolution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeDiscreteSpace:
    n = 4


class FakeContinuousSpace:
    shape = (2,)
    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])


def build_cartpole_tree():
    """Manually build a CartPole policy tree matching Figure 4 of the paper."""
    # IF s[2] <= 0.0 (pole angle):
    #   IF s[3] <= 0.0 (pole velocity): RETURN 0 (push left)
    #   ELSE:                            RETURN 1 (push right)
    # ELSE:
    #   RETURN 1 (push right)
    leaf_ll = LeafNode(0)
    leaf_lr = LeafNode(1)
    inner_left = DecisionNode(3, 0.0)
    inner_left.left = leaf_ll
    inner_left.right = leaf_lr

    leaf_r = LeafNode(1)
    root = DecisionNode(2, 0.0)
    root.left = inner_left
    root.right = leaf_r

    return DecisionTreePolicy(root, state_dim=4, action_space=FakeDiscreteSpace())


# ---------------------------------------------------------------------------
# Decision tree tests
# ---------------------------------------------------------------------------


class TestDecisionTreePolicy:
    def test_act_goes_left(self):
        """A state satisfying root condition should go left."""
        tree = build_cartpole_tree()
        # s[2] = -0.1 <= 0.0 → go left; s[3] = -0.5 <= 0.0 → action 0
        state = np.array([0.0, 0.0, -0.1, -0.5])
        assert tree.act(state) == 0

    def test_act_goes_right_from_root(self):
        """A state failing root condition should go right."""
        tree = build_cartpole_tree()
        # s[2] = 0.3 > 0.0 → right leaf → action 1
        state = np.array([0.0, 0.0, 0.3, 0.0])
        assert tree.act(state) == 1

    def test_depth(self):
        tree = build_cartpole_tree()
        assert tree.get_depth() == 2

    def test_num_rules(self):
        tree = build_cartpole_tree()
        assert tree.get_num_rules() == 3

    def test_features_used(self):
        tree = build_cartpole_tree()
        assert tree.get_features_used() == {2, 3}

    def test_to_rules_count(self):
        tree = build_cartpole_tree()
        rules = tree.to_rules()
        assert len(rules) == 3

    def test_summary_keys(self):
        tree = build_cartpole_tree()
        s = tree.summary()
        for key in ("depth", "num_rules", "features_used", "state_dim"):
            assert key in s

    def test_single_leaf_policy(self):
        """A constant policy (single leaf) always returns the same action."""
        tree = DecisionTreePolicy(LeafNode(2), state_dim=4, action_space=FakeDiscreteSpace())
        for _ in range(10):
            state = np.random.randn(4)
            assert tree.act(state) == 2


# ---------------------------------------------------------------------------
# Grammatical Evolution tests
# ---------------------------------------------------------------------------


class TestGrammaticalEvolution:
    def test_random_genotype_shape(self):
        ge = GrammaticalEvolution(
            state_dim=4,
            action_space=FakeDiscreteSpace(),
            max_depth=3,
            genotype_length=80,
        )
        g = ge.random_genotype()
        assert g.shape == (80,)
        assert g.dtype == np.int32

    def test_decode_returns_policy_or_none(self):
        ge = GrammaticalEvolution(
            state_dim=4,
            action_space=FakeDiscreteSpace(),
            max_depth=3,
            genotype_length=80,
        )
        for _ in range(20):
            g = ge.random_genotype()
            result = ge.decode(g)
            assert result is None or isinstance(result, DecisionTreePolicy)

    def test_decoded_policy_acts(self):
        """Decoded policy should produce valid discrete actions."""
        ge = GrammaticalEvolution(
            state_dim=4,
            action_space=FakeDiscreteSpace(),
            max_depth=3,
            genotype_length=80,
        )
        state = np.zeros(4)
        valid = 0
        for _ in range(50):
            g = ge.random_genotype()
            policy = ge.decode(g)
            if policy is not None:
                action = policy.act(state)
                assert 0 <= action < FakeDiscreteSpace.n
                valid += 1
        # At least some genotypes should decode successfully
        assert valid > 0

    def test_depth_constraint(self):
        """Decoded tree should never exceed max_depth."""
        ge = GrammaticalEvolution(
            state_dim=4,
            action_space=FakeDiscreteSpace(),
            max_depth=3,
            genotype_length=80,
        )
        for _ in range(30):
            g = ge.random_genotype()
            policy = ge.decode(g)
            if policy is not None:
                assert policy.get_depth() <= 3

    def test_crossover_preserves_length(self):
        ge = GrammaticalEvolution(
            state_dim=4,
            action_space=FakeDiscreteSpace(),
            max_depth=3,
            genotype_length=80,
        )
        g1 = ge.random_genotype()
        g2 = ge.random_genotype()
        c1, c2 = ge.crossover_two_point(g1, g2)
        assert len(c1) == 80
        assert len(c2) == 80

    def test_mutation_changes_genotype(self):
        ge = GrammaticalEvolution(
            state_dim=4,
            action_space=FakeDiscreteSpace(),
            max_depth=3,
            genotype_length=200,
        )
        g = ge.random_genotype()
        mutated = ge.mutate(g, mutation_prob=0.5)  # high probability for testing
        # With prob 0.5 per codon and 200 codons, extremely unlikely to be identical
        assert not np.array_equal(g, mutated)

    def test_continuous_action_shape(self):
        """Decoded policy for continuous actions should return array of correct shape."""
        ge = GrammaticalEvolution(
            state_dim=4,
            action_space=FakeContinuousSpace(),
            max_depth=3,
            genotype_length=100,
        )
        state = np.zeros(4)
        found = False
        for _ in range(50):
            g = ge.random_genotype()
            policy = ge.decode(g)
            if policy is not None:
                action = policy.act(state)
                assert hasattr(action, "__len__"), "Continuous action should be array-like"
                assert len(action) == 2
                found = True
                break
        assert found, "No valid policy decoded for continuous action space"

    def test_continuous_action_bounds(self):
        """Decoded continuous actions should lie within action bounds."""
        ge = GrammaticalEvolution(
            state_dim=4,
            action_space=FakeContinuousSpace(),
            max_depth=3,
            genotype_length=100,
        )
        state = np.zeros(4)
        for _ in range(50):
            g = ge.random_genotype()
            policy = ge.decode(g)
            if policy is not None:
                action = policy.act(state)
                assert np.all(action >= -1.0 - 1e-9)
                assert np.all(action <= 1.0 + 1e-9)
