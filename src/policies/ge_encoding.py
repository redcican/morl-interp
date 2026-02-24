"""
Grammatical Evolution (GE) encoding for decision tree policies.

GE maps an integer genotype (a fixed-length vector of codons) to a decision
tree phenotype via a context-free grammar. When expanding a non-terminal that
has k production rules, codon c selects rule c mod k. Wrap-around is used if
the genotype is exhausted before derivation completes.

Grammar (simplified):
    <tree>   ::= <node>
    <node>   ::= INTERNAL(<feat>, <thresh>, <node_left>, <node_right>)
               | LEAF(<action>)
    <feat>   ::= 0 | 1 | ... | state_dim - 1
    <thresh> ::= continuous float derived from codon pair
    <action> ::= 0 | 1 | ... | n_actions - 1          (discrete)
               | [a_0, ..., a_{d-1}]                   (continuous)

The maximum derivation depth is enforced by forcing leaf production at depth
d_max, which corresponds to Assumption 1 (Bounded Policy Space) in the paper.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple, Union

import numpy as np

from .decision_tree import DecisionNode, DecisionTreePolicy, LeafNode


class GrammaticalEvolution:
    """
    Grammatical Evolution encoder/decoder for binary decision tree policies.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the environment's observation space.
    action_space : gymnasium.Space
        The environment's action space (discrete or box).
    max_depth : int
        Maximum tree depth enforced during derivation (Assumption 1 in paper).
    genotype_length : int
        Fixed length of the integer genotype vector.
    codon_range : int
        Each codon is drawn from [0, codon_range). Default 256.
    """

    # Number of production alternatives for <node>:
    # 0 = internal (DecisionNode), 1 = leaf (LeafNode)
    _N_NODE_RULES = 2

    def __init__(
        self,
        state_dim: int,
        action_space,
        max_depth: int = 5,
        genotype_length: int = 100,
        codon_range: int = 256,
    ) -> None:
        self.state_dim = state_dim
        self.action_space = action_space
        self.max_depth = max_depth
        self.genotype_length = genotype_length
        self.codon_range = codon_range

        # Determine action type
        if hasattr(action_space, "n"):
            self.is_discrete = True
            self.n_actions = int(action_space.n)
        else:
            self.is_discrete = False
            self.n_actions = int(action_space.shape[0])
            self.action_low = action_space.low.astype(float)
            self.action_high = action_space.high.astype(float)

    # ------------------------------------------------------------------
    # Decoding: genotype → phenotype
    # ------------------------------------------------------------------

    def decode(self, genotype: np.ndarray) -> Optional[DecisionTreePolicy]:
        """
        Decode an integer genotype into a DecisionTreePolicy.

        Returns None if the genotype fails to produce a valid tree (e.g.,
        recursion depth exceeded or structural inconsistency).
        """
        self._codons = genotype.tolist()
        self._pos = 0

        try:
            root = self._expand_node(depth=0)
        except (RecursionError, ValueError):
            return None

        if root is None:
            return None

        return DecisionTreePolicy(root, self.state_dim, self.action_space)

    def _next_codon(self) -> int:
        """Read the next codon with wrap-around."""
        val = int(self._codons[self._pos % len(self._codons)])
        self._pos += 1
        return val

    def _expand_node(
        self, depth: int
    ) -> Optional[Union[DecisionNode, LeafNode]]:
        """Expand the <node> non-terminal at the given tree depth."""
        if depth >= self.max_depth:
            # Grammar forces leaf production at maximum depth
            return self._expand_leaf()

        # Choose production: 0 → internal node, 1 → leaf
        choice = self._next_codon() % self._N_NODE_RULES
        if choice == 0:
            return self._expand_internal(depth)
        return self._expand_leaf()

    def _expand_internal(
        self, depth: int
    ) -> Optional[DecisionNode]:
        """Expand an internal DecisionNode."""
        # Feature index
        feat_idx = self._next_codon() % self.state_dim

        # Threshold: two codons encode a float in [-5, 5].
        # The first codon provides the integer part shift and the second
        # the fractional part, giving resolution of 0.01 over 10 units.
        c_int = self._next_codon() % 10          # 0..9  → integer offset
        c_frac = self._next_codon() % 100        # 0..99 → fractional part
        threshold = float(c_int) + float(c_frac) / 100.0 - 5.0  # ∈ [-5, 4.99]

        node = DecisionNode(feat_idx, threshold)
        node.left = self._expand_node(depth + 1)
        node.right = self._expand_node(depth + 1)

        if node.left is None or node.right is None:
            return None
        return node

    def _expand_leaf(self) -> LeafNode:
        """Expand a LeafNode with a discrete or continuous action."""
        if self.is_discrete:
            action = self._next_codon() % self.n_actions
        else:
            # Each action dimension encoded by one codon mapped to [low, high]
            action = np.empty(self.n_actions, dtype=float)
            for i in range(self.n_actions):
                c = self._next_codon() % 1000
                action[i] = self.action_low[i] + (
                    c / 999.0 * (self.action_high[i] - self.action_low[i])
                )
        return LeafNode(action)

    # ------------------------------------------------------------------
    # Genotype generation and genetic operators
    # ------------------------------------------------------------------

    def random_genotype(self) -> np.ndarray:
        """Sample a random integer genotype."""
        return np.random.randint(
            0, self.codon_range, size=self.genotype_length, dtype=np.int32
        )

    def crossover_two_point(
        self, g1: np.ndarray, g2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover, preserving genotype length."""
        n = len(g1)
        pts = sorted(random.sample(range(1, n), 2))
        p, q = pts
        c1 = np.concatenate([g1[:p], g2[p:q], g1[q:]])
        c2 = np.concatenate([g2[:p], g1[p:q], g2[q:]])
        return c1.astype(np.int32), c2.astype(np.int32)

    def crossover_uniform(
        self, g1: np.ndarray, g2: np.ndarray, prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover with swap probability prob."""
        mask = np.random.random(len(g1)) < prob
        c1, c2 = g1.copy(), g2.copy()
        c1[mask], c2[mask] = g2[mask], g1[mask]
        return c1.astype(np.int32), c2.astype(np.int32)

    def mutate(
        self, genotype: np.ndarray, mutation_prob: float = 0.01
    ) -> np.ndarray:
        """
        Per-codon uniform mutation: each codon is replaced with probability
        mutation_prob by a new random codon drawn from [0, codon_range).
        """
        g = genotype.copy()
        mask = np.random.random(len(g)) < mutation_prob
        g[mask] = np.random.randint(0, self.codon_range, size=int(mask.sum()))
        return g.astype(np.int32)
