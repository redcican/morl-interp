"""
MOEA/D optimizer for multi-objective interpretable policy search.

MOEA/D (Zhang & Li, 2007) decomposes the multi-objective problem into N
scalar subproblems using weight vectors. Each subproblem is optimized
collaboratively with its neighbors via a Tchebycheff aggregation function.

The Tchebycheff approach handles both convex and non-convex Pareto fronts,
making it well-suited for performance–interpretability trade-offs whose
front shape is a priori unknown.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..evaluation.pareto_metrics import compute_all_metrics, non_dominated_sort
from .morl_framework import MORLFramework

logger = logging.getLogger(__name__)


def tchebycheff(
    objectives: np.ndarray,
    weight: np.ndarray,
    ideal: np.ndarray,
) -> float:
    """
    Tchebycheff aggregation (to minimize): max_i w_i * |z*_i - f_i(x)|.

    For maximization objectives, the ideal point z* is the element-wise
    maximum seen so far, so we want the Tchebycheff value to be small.
    """
    return float(np.max(weight * np.abs(ideal - objectives)))


class MOEADOptimizer:
    """
    MOEA/D optimizer for interpretable policy Pareto front discovery.

    Parameters
    ----------
    framework : MORLFramework providing evaluation and GE encoding.
    """

    def __init__(self, framework: MORLFramework) -> None:
        self.framework = framework
        self.config = framework.config

        moead_cfg = self.config.get("moead", {})
        self.n_subproblems: int = self.config["ea"]["population_size"]
        self.neighborhood_size: int = moead_cfg.get("neighborhood_size", 10)
        self.n_obj: int = 2  # performance + interpretability

        # Uniformly distributed weight vectors for 2 objectives
        self.weights = self._uniform_weights(self.n_subproblems)

        # Pre-compute T-nearest weight neighborhoods
        self.neighborhoods = self._build_neighborhoods()

        # Genotype operator parameters
        ge_cfg = self.config["ge"]
        self.mut_prob: float = ge_cfg.get("mutation_prob", 0.01)
        self.mut_indiv_prob: float = ge_cfg.get("mutation_indiv_prob", 0.2)

    # ------------------------------------------------------------------
    # Weight vector utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _uniform_weights(n: int) -> np.ndarray:
        """Generate n uniformly spaced 2-D weight vectors on the unit simplex."""
        weights = np.zeros((n, 2), dtype=float)
        for i in range(n):
            w = i / max(n - 1, 1)
            weights[i] = [w, 1.0 - w]
        # Avoid degenerate all-zero weights by clipping away from zero
        weights = np.clip(weights, 1e-3, 1.0)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def _build_neighborhoods(self) -> List[List[int]]:
        """Return, for each subproblem, the indices of its T nearest neighbors."""
        T = min(self.neighborhood_size, self.n_subproblems)
        neighborhoods = []
        for i in range(self.n_subproblems):
            dists = np.linalg.norm(self.weights - self.weights[i], axis=1)
            neighbors = np.argsort(dists)[:T].tolist()
            neighborhoods.append(neighbors)
        return neighborhoods

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _crossover(
        self, g1: np.ndarray, g2: np.ndarray
    ) -> np.ndarray:
        """Uniform crossover between two integer genotypes."""
        mask = np.random.random(len(g1)) < 0.5
        child = g1.copy()
        child[mask] = g2[mask]
        return child

    def _mutate(self, genotype: np.ndarray) -> np.ndarray:
        """Per-codon uniform integer mutation."""
        return self.framework.ge.mutate(genotype, self.mut_prob)

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def run(self) -> List[Dict]:
        """
        Run MOEA/D and return all non-dominated solutions.

        Returns
        -------
        List of result dictionaries with 'genotype', 'policy', 'performance',
        'interpretability', 'objectives', 'tree_summary'.
        """
        n_gen = self.config["ea"]["generations"]

        # --- Initialization ---
        population = [self.framework.ge.random_genotype()
                      for _ in range(self.n_subproblems)]
        objectives = [self._eval(g) for g in population]

        # Ideal point: element-wise maximum observed so far
        ideal = np.max(objectives, axis=0)

        logger.info(
            "MOEA/D | env=%s | subproblems=%d | gen=%d",
            self.framework.env_spec.env_id,
            self.n_subproblems,
            n_gen,
        )

        # --- Generational loop ---
        for gen in range(n_gen):
            for i in range(self.n_subproblems):
                neighbors = self.neighborhoods[i]

                # Select two parents from the neighborhood
                p_idx = np.random.choice(neighbors, 2, replace=False)
                child = self._crossover(population[p_idx[0]], population[p_idx[1]])

                # Apply mutation with individual probability
                if np.random.random() < self.mut_indiv_prob:
                    child = self._mutate(child)

                child_obj = self._eval(child)

                # Update ideal point
                ideal = np.maximum(ideal, child_obj)

                # Update neighborhood solutions using Tchebycheff criterion
                for j in neighbors:
                    curr_fit = tchebycheff(objectives[j], self.weights[j], ideal)
                    child_fit = tchebycheff(child_obj, self.weights[j], ideal)
                    if child_fit <= curr_fit:
                        population[j] = child.copy()
                        objectives[j] = child_obj.copy()

            # --- Logging ---
            obj_arr = np.array(objectives)
            nd_idx = non_dominated_sort(obj_arr)
            front_obj = obj_arr[nd_idx]
            metrics = compute_all_metrics(
                front_obj,
                true_front=self.framework.env_spec.true_pareto_front,
            )

            self.framework.history.append(
                {"generation": gen + 1, "pareto_size": len(nd_idx), **metrics}
            )

            if (gen + 1) % max(1, n_gen // 10) == 0:
                logger.info(
                    "  Gen %4d/%d | front=%d | HV=%.4f | S=%.4f",
                    gen + 1,
                    n_gen,
                    metrics["front_size"],
                    metrics["hypervolume"],
                    metrics["sparsity"],
                )

        # --- Extract final Pareto front ---
        obj_arr = np.array(objectives)
        nd_idx = non_dominated_sort(obj_arr)

        results = []
        for i in nd_idx:
            genotype = population[i]
            obj_tuple = (float(objectives[i][0]), float(objectives[i][1]))
            result = self.framework.record_result(genotype, obj_tuple)
            results.append(result)

        self.framework.pareto_front = results
        logger.info(
            "MOEA/D completed | final Pareto front size=%d", len(results)
        )
        return results

    def _eval(self, genotype: np.ndarray) -> np.ndarray:
        """Evaluate genotype and return [performance, interpretability]."""
        perf, interp, _ = self.framework.evaluate(genotype)
        return np.array([perf, interp], dtype=float)
