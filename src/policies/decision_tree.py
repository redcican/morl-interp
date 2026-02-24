"""
Decision tree policy representation for interpretable reinforcement learning.

A policy is a binary decision tree where internal nodes test a single state
feature against a threshold, and leaf nodes return actions. The policy is
deterministic: given a state, it traverses the tree from root to the
appropriate leaf and returns that leaf's action.
"""

from __future__ import annotations

from typing import List, Optional, Set, Union
import numpy as np


class DecisionNode:
    """Internal node that tests s[feature_idx] <= threshold."""

    def __init__(self, feature_idx: int, threshold: float) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left: Optional[Union[DecisionNode, LeafNode]] = None   # True branch
        self.right: Optional[Union[DecisionNode, LeafNode]] = None  # False branch

    def __repr__(self) -> str:
        return f"DecisionNode(s[{self.feature_idx}] <= {self.threshold:.4f})"


class LeafNode:
    """Leaf node that returns a fixed action."""

    def __init__(self, action: Union[int, np.ndarray]) -> None:
        self.action = action

    def __repr__(self) -> str:
        return f"LeafNode(action={self.action})"


class DecisionTreePolicy:
    """
    A decision tree policy for reinforcement learning.

    The tree maps states to actions by traversing from root to leaf:
    - At each internal node, if s[feature_idx] <= threshold, go left; else go right.
    - At each leaf, return the stored action.
    """

    def __init__(
        self,
        root: Union[DecisionNode, LeafNode],
        state_dim: int,
        action_space,
    ) -> None:
        self.root = root
        self.state_dim = state_dim
        self.action_space = action_space

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """Return the action for the given state by traversing the tree."""
        node = self.root
        while isinstance(node, DecisionNode):
            if state[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.action

    # ------------------------------------------------------------------
    # Structural properties
    # ------------------------------------------------------------------

    def get_depth(self) -> int:
        """Maximum root-to-leaf path length (a single-node tree has depth 0)."""
        return self._depth(self.root)

    def _depth(self, node: Union[DecisionNode, LeafNode]) -> int:
        if isinstance(node, LeafNode):
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def get_leaves(self) -> List[LeafNode]:
        """Return all leaf nodes."""
        leaves: List[LeafNode] = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(
        self,
        node: Union[DecisionNode, LeafNode],
        leaves: List[LeafNode],
    ) -> None:
        if isinstance(node, LeafNode):
            leaves.append(node)
        else:
            self._collect_leaves(node.left, leaves)
            self._collect_leaves(node.right, leaves)

    def get_num_rules(self) -> int:
        """Number of decision rules = number of leaves."""
        return len(self.get_leaves())

    def get_features_used(self) -> Set[int]:
        """Set of state feature indices tested by internal nodes."""
        features: Set[int] = set()
        self._collect_features(self.root, features)
        return features

    def _collect_features(
        self,
        node: Union[DecisionNode, LeafNode],
        features: Set[int],
    ) -> None:
        if isinstance(node, DecisionNode):
            features.add(node.feature_idx)
            self._collect_features(node.left, features)
            self._collect_features(node.right, features)

    # ------------------------------------------------------------------
    # Human-readable output
    # ------------------------------------------------------------------

    def to_rules(self) -> List[str]:
        """Return human-readable if-then rules for each leaf."""
        rules: List[str] = []
        self._extract_rules(self.root, [], rules)
        return rules

    def _extract_rules(
        self,
        node: Union[DecisionNode, LeafNode],
        conditions: List[str],
        rules: List[str],
    ) -> None:
        if isinstance(node, LeafNode):
            body = " AND ".join(conditions) if conditions else "TRUE"
            rules.append(f"IF {body} THEN action = {node.action}")
        else:
            feat = f"s[{node.feature_idx}]"
            thresh = f"{node.threshold:.4f}"
            self._extract_rules(
                node.left,
                conditions + [f"{feat} <= {thresh}"],
                rules,
            )
            self._extract_rules(
                node.right,
                conditions + [f"{feat} >  {thresh}"],
                rules,
            )

    def print_tree(self, indent: int = 0) -> None:
        """Pretty-print the tree structure."""
        self._print_node(self.root, indent)

    def _print_node(
        self,
        node: Union[DecisionNode, LeafNode],
        indent: int,
    ) -> None:
        prefix = "  " * indent
        if isinstance(node, LeafNode):
            print(f"{prefix}RETURN {node.action}")
        else:
            print(f"{prefix}IF s[{node.feature_idx}] <= {node.threshold:.4f}:")
            self._print_node(node.left, indent + 1)
            print(f"{prefix}ELSE:")
            self._print_node(node.right, indent + 1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dictionary of structural properties."""
        return {
            "depth": self.get_depth(),
            "num_rules": self.get_num_rules(),
            "features_used": sorted(self.get_features_used()),
            "num_features_used": len(self.get_features_used()),
            "state_dim": self.state_dim,
        }
