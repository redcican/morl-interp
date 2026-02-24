"""
Weighted sum scalarization baseline.

Combines performance and interpretability into a single objective:
    J_total = alpha * J_perf + (1 - alpha) * J_interp

for a set of alpha values in [0, 1]. Each alpha value is optimized
independently using single-objective evolutionary search, and the
resulting policies form the weighted-sum approximate Pareto front.

Limitation: linear scalarization cannot recover non-convex regions of
the Pareto front (Deb & Jain, 2013), unlike NSGA-III and MOEA/D.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ..algorithms.morl_framework import MORLFramework

logger = logging.getLogger(__name__)

# Default alpha grid used in the paper (Section 6.1)
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


class WeightedSumBaseline:
    """
    Weighted sum scalarization for performance–interpretability trade-off.

    For each alpha, runs a simple evolutionary strategy maximizing
    alpha * performance + (1 - alpha) * interpretability.

    Parameters
    ----------
    framework : MORLFramework providing evaluation and GE encoding.
    alphas    : List of alpha values to sweep over.
    """

    def __init__(
        self,
        framework: MORLFramework,
        alphas: List[float] = None,
    ) -> None:
        self.framework = framework
        self.config = framework.config
        self.alphas = alphas if alphas is not None else DEFAULT_ALPHAS

    def _scalar_fitness(
        self, genotype: np.ndarray, alpha: float
    ) -> float:
        """Evaluate the weighted sum objective for one genotype."""
        perf, interp, _ = self.framework.evaluate(genotype)
        return alpha * perf + (1.0 - alpha) * interp

    def _optimize_alpha(self, alpha: float) -> Dict:
        """
        Run a (mu + lambda) evolutionary strategy for one alpha value.
        Returns the best-found policy and its objectives.
        """
        cfg = self.config
        pop_size = cfg["ea"]["population_size"]
        n_gen = cfg["ea"]["generations"]
        mut_prob = cfg["ge"].get("mutation_prob", 0.01)
        cx_prob = cfg["ge"].get("crossover_prob", 0.7)

        # Initialize population
        population = [self.framework.ge.random_genotype() for _ in range(pop_size)]
        fitness = np.array([self._scalar_fitness(g, alpha) for g in population])

        best_idx = int(np.argmax(fitness))
        best_g = population[best_idx].copy()
        best_f = float(fitness[best_idx])

        for gen in range(n_gen):
            # Tournament selection + crossover + mutation
            offspring = []
            for _ in range(pop_size):
                # Binary tournament
                i1, i2 = np.random.choice(pop_size, 2, replace=False)
                parent1 = population[i1 if fitness[i1] >= fitness[i2] else i2]
                i3, i4 = np.random.choice(pop_size, 2, replace=False)
                parent2 = population[i3 if fitness[i3] >= fitness[i4] else i4]

                # Crossover
                if np.random.random() < cx_prob:
                    child, _ = self.framework.ge.crossover_two_point(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                child = self.framework.ge.mutate(child, mut_prob)
                offspring.append(child)

            # Evaluate offspring
            off_fitness = np.array([self._scalar_fitness(g, alpha) for g in offspring])

            # (mu + lambda) replacement
            combined = population + offspring
            combined_f = np.concatenate([fitness, off_fitness])
            top_idx = np.argsort(-combined_f)[:pop_size]
            population = [combined[i] for i in top_idx]
            fitness = combined_f[top_idx]

            if fitness[0] > best_f:
                best_f = float(fitness[0])
                best_g = population[0].copy()

        # Evaluate the best genotype on both objectives
        perf, interp, components = self.framework.evaluate(best_g)
        return self.framework.record_result(best_g, (perf, interp), components)

    def run(self) -> List[Dict]:
        """
        Optimize for each alpha value and return the collection of best policies.

        Returns
        -------
        List of result dictionaries, one per alpha value.
        """
        results = []
        for alpha in self.alphas:
            logger.info("WeightedSum | alpha=%.2f | optimizing...", alpha)
            result = self._optimize_alpha(alpha)
            result["alpha"] = alpha
            results.append(result)
            logger.info(
                "  alpha=%.2f | perf=%.3f | interp=%.3f",
                alpha,
                result["performance"],
                result["interpretability"],
            )
        return results
