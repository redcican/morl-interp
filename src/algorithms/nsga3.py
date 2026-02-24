"""
NSGA-III optimizer for multi-objective interpretable policy search.

NSGA-III (Deb & Jain, 2014) uses reference point-guided selection to
maintain a diverse population along the Pareto front. It extends NSGA-II
by replacing the crowding distance with structured reference directions
on a unit simplex, which preserves a well-distributed approximation front.

This implementation uses DEAP for the evolutionary infrastructure
and integrates with MORLFramework for policy evaluation.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from ..evaluation.pareto_metrics import compute_all_metrics, non_dominated_sort
from .morl_framework import MORLFramework

logger = logging.getLogger(__name__)

# Guard against re-registering DEAP types across runs
_CREATOR_INITIALIZED = False


def _init_creator() -> None:
    global _CREATOR_INITIALIZED
    if not _CREATOR_INITIALIZED:
        creator.create("FitnessMultiMax", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMultiMax)
        _CREATOR_INITIALIZED = True


class NSGA3Optimizer:
    """
    NSGA-III based multi-objective evolutionary search for interpretable
    decision tree policies.

    Parameters
    ----------
    framework : MORLFramework instance providing evaluation and GE encoding.
    """

    def __init__(self, framework: MORLFramework) -> None:
        self.framework = framework
        self.config = framework.config
        _init_creator()
        self._toolbox = self._build_toolbox()

    # ------------------------------------------------------------------
    # DEAP toolbox configuration
    # ------------------------------------------------------------------

    def _build_toolbox(self) -> base.Toolbox:
        ge_cfg = self.config["ge"]
        geno_len = ge_cfg["genotype_length"]
        n3_cfg = self.config.get("nsga3", {})
        n_div = n3_cfg.get("reference_divisions", 12)

        toolbox = base.Toolbox()

        # Genotype factory: list of integers in [0, 255]
        toolbox.register("attr_int", random.randint, 0, 255)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_int,
            n=geno_len,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation
        framework = self.framework

        def evaluate(individual: list) -> Tuple[float, float]:
            genotype = np.array(individual, dtype=np.int32)
            perf, interp, _ = framework.evaluate(genotype)
            return float(perf), float(interp)

        toolbox.register("evaluate", evaluate)

        # Genetic operators
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=255,
            indpb=ge_cfg.get("mutation_prob", 0.01),
        )

        # Reference-point-based selection (NSGA-III)
        ref_points = tools.uniform_reference_points(nobj=2, p=n_div)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

        return toolbox

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def run(self) -> List[Dict]:
        """
        Run NSGA-III and return the final Pareto front.

        Returns
        -------
        List of result dictionaries, one per non-dominated solution, each
        containing 'genotype', 'policy', 'performance', 'interpretability',
        'objectives', 'tree_summary'.
        """
        config = self.config
        pop_size = config["ea"]["population_size"]
        n_gen = config["ea"]["generations"]
        cx_prob = config["ge"].get("crossover_prob", 0.7)
        mut_prob = config["ge"].get("mutation_indiv_prob", 0.2)

        toolbox = self._toolbox

        # --- Initialization ---
        population = toolbox.population(n=pop_size)
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        logger.info(
            "NSGA-III | env=%s | pop=%d | gen=%d",
            self.framework.env_spec.env_id,
            pop_size,
            n_gen,
        )

        # --- Generational loop ---
        for gen in range(n_gen):
            offspring = algorithms.varAnd(population, toolbox, cx_prob, mut_prob)

            # Evaluate new individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # NSGA-III selection over combined pool
            population = toolbox.select(population + offspring, pop_size)

            # Logging
            front = tools.sortNondominated(
                population, len(population), first_front_only=True
            )[0]
            front_obj = np.array([ind.fitness.values for ind in front])
            metrics = compute_all_metrics(
                front_obj,
                true_front=self.framework.env_spec.true_pareto_front,
            )

            self.framework.history.append(
                {
                    "generation": gen + 1,
                    "pareto_size": len(front),
                    **metrics,
                }
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
        final_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )[0]

        results = []
        for ind in final_front:
            genotype = np.array(ind, dtype=np.int32)
            objectives = ind.fitness.values
            result = self.framework.record_result(genotype, objectives)
            results.append(result)

        self.framework.pareto_front = results
        logger.info(
            "NSGA-III completed | final Pareto front size=%d", len(results)
        )
        return results
