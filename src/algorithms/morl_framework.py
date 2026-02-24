"""
Base evaluation framework for Multi-Objective Evolutionary RL.

MORLFramework handles:
- Environment management and policy evaluation
- Interpretability metric computation
- Genotype ↔ phenotype conversion via GrammaticalEvolution
- Result storage

Both NSGA-III and MOEA/D inherit from this class and implement run().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..environments.wrappers import collect_trajectories, get_env_spec, make_env
from ..metrics.interpretability import compute_composite_interpretability
from ..policies.decision_tree import DecisionTreePolicy
from ..policies.ge_encoding import GrammaticalEvolution

logger = logging.getLogger(__name__)


class MORLFramework:
    """
    Base class for multi-objective evolutionary RL experiments.

    Parameters
    ----------
    config : Experiment configuration dictionary loaded from YAML.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.env_spec = get_env_spec(config)

        ea_cfg = config["ea"]
        self.pop_size: int = ea_cfg["population_size"]
        self.max_generations: int = ea_cfg["generations"]
        self.n_episodes: int = ea_cfg["episodes_per_eval"]
        self.discount: float = ea_cfg["discount"]
        self.d_max: int = ea_cfg["max_tree_depth"]
        self.d_min: float = 1.0
        self.r_max: float = float(2 ** self.d_max)
        self.r_min: float = 2.0

        iw = config["interpretability_weights"]
        self.interp_weights: Tuple[float, float, float, float] = (
            iw["w1"],
            iw["w2"],
            iw["w3"],
            iw["w4"],
        )

        # Grammatical Evolution encoder
        env_tmp = make_env(config)
        self.ge = GrammaticalEvolution(
            state_dim=self.env_spec.state_dim,
            action_space=env_tmp.action_space,
            max_depth=self.d_max,
            genotype_length=config["ge"]["genotype_length"],
        )
        env_tmp.close()

        # Results populated by run()
        self.pareto_front: List[Dict] = []
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Policy evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        genotype: np.ndarray,
        seed: Optional[int] = None,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Decode a genotype and evaluate the resulting policy.

        Returns
        -------
        performance   : Scalar task performance (discounted return or sum
                        over objectives for MORL environments).
        interpretability : Composite interpretability score I(T) ∈ [0, 1].
        components    : Dictionary of individual metric values.
        """
        policy = self.decode(genotype)
        if policy is None:
            return -1000.0, 0.0, {}

        perf, action_trajs, vec_reward = collect_trajectories(
            policy=policy,
            config=self.config,
            n_episodes=self.n_episodes,
            discount=self.discount,
            seed=seed,
        )

        # For MORL environments, use the sum of objective averages as scalar
        # performance (for single-objective use the scalar directly).
        if vec_reward is not None:
            perf = float(vec_reward.sum())

        interp, components = compute_composite_interpretability(
            tree=policy,
            trajectories=action_trajs,
            state_dim=self.env_spec.state_dim,
            weights=self.interp_weights,
            d_min=self.d_min,
            d_max=float(self.d_max),
            r_min=self.r_min,
            r_max=self.r_max,
            is_continuous=not self.env_spec.is_discrete,
            action_dim=self.env_spec.n_actions,
            a_max=self.env_spec.a_max,
        )

        return float(perf), float(interp), components

    def evaluate_vec(
        self,
        genotype: np.ndarray,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        Evaluate and return the full objective vector for MORL environments.

        Returns
        -------
        vec_reward    : Average reward vector across episodes.
        interpretability : Composite interpretability score.
        components    : Individual metric values.
        """
        policy = self.decode(genotype)
        if policy is None:
            n = 2  # default to 2 task objectives
            return np.full(n, -1000.0), 0.0, {}

        _, action_trajs, vec_reward = collect_trajectories(
            policy=policy,
            config=self.config,
            n_episodes=self.n_episodes,
            discount=self.discount,
            seed=seed,
        )

        if vec_reward is None:
            # Fallback for single-objective env
            vec_reward = np.array([0.0])

        interp, components = compute_composite_interpretability(
            tree=policy,
            trajectories=action_trajs,
            state_dim=self.env_spec.state_dim,
            weights=self.interp_weights,
            d_min=self.d_min,
            d_max=float(self.d_max),
            r_min=self.r_min,
            r_max=self.r_max,
            is_continuous=not self.env_spec.is_discrete,
            action_dim=self.env_spec.n_actions,
            a_max=self.env_spec.a_max,
        )

        return vec_reward, float(interp), components

    # ------------------------------------------------------------------
    # Genotype utilities
    # ------------------------------------------------------------------

    def decode(self, genotype: np.ndarray) -> Optional[DecisionTreePolicy]:
        """Decode integer genotype to DecisionTreePolicy (or None on failure)."""
        return self.ge.decode(genotype)

    def random_population(self) -> List[np.ndarray]:
        """Return a list of pop_size random genotypes."""
        return [self.ge.random_genotype() for _ in range(self.pop_size)]

    # ------------------------------------------------------------------
    # Result storage
    # ------------------------------------------------------------------

    def record_result(
        self,
        genotype: np.ndarray,
        objectives: Tuple[float, float],
        components: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Build a result dictionary for storage."""
        policy = self.decode(genotype)
        result = {
            "genotype": genotype.copy(),
            "policy": policy,
            "performance": float(objectives[0]),
            "interpretability": float(objectives[1]),
            "objectives": list(objectives),
        }
        if components:
            result["components"] = components
        if policy is not None:
            result["tree_summary"] = policy.summary()
        return result

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def run(self) -> List[Dict]:
        """Execute the evolutionary optimization. Subclasses must implement."""
        raise NotImplementedError
