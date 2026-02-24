# Multi-Objective Evolutionary Reinforcement Learning for Pareto-Optimal Interpretable Policies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official implementation of the paper:

> **Multi-Objective Evolutionary Reinforcement Learning for Pareto-Optimal Interpretable Policies**
> Shikun Chen, Yangguang Liu


## Overview

This repository provides a bi-objective evolutionary reinforcement learning framework that treats **policy interpretability as an explicit optimization objective** alongside task performance. Rather than post-hoc explanation of black-box agents, we co-evolve decision tree policies for both objectives using NSGA-III and MOEA/D, discovering a Pareto front of policies that let practitioners choose their preferred performance–interpretability trade-off.

**Key features:**
- Decision tree policies encoded via Grammatical Evolution (GE)
- Composite interpretability proxy metric: depth, rule count, temporal coherence, causal sufficiency
- NSGA-III and MOEA/D optimization over the two-objective space
- Evaluation on four environments: Deep Sea Treasure, MO-HalfCheetah, CartPole, Hopper
- Baselines: weighted sum scalarization, VIPER, PIRL, PGMORL, PSL-MORL, PPO

## Repository Structure

```
Interpretable_RL_code/
├── src/
│   ├── policies/
│   │   ├── decision_tree.py      # Decision tree policy representation
│   │   └── ge_encoding.py        # Grammatical Evolution encoder/decoder
│   ├── metrics/
│   │   └── interpretability.py   # M_depth, M_rules, M_temporal, M_causal
│   ├── algorithms/
│   │   ├── morl_framework.py     # Base evaluation framework
│   │   ├── nsga3.py              # NSGA-III optimizer
│   │   └── moead.py              # MOEA/D optimizer
│   ├── environments/
│   │   └── wrappers.py           # Environment factory and trajectory collection
│   ├── baselines/
│   │   ├── weighted_sum.py       # Weighted sum scalarization baseline
│   │   └── viper.py              # VIPER policy extraction baseline
│   └── evaluation/
│       └── pareto_metrics.py     # Hypervolume, sparsity, coverage metrics
├── configs/
│   ├── cartpole.yaml             # CartPole experiment configuration
│   ├── dst.yaml                  # Deep Sea Treasure configuration
│   ├── hopper.yaml               # Hopper configuration
│   └── mo_halfcheetah.yaml       # MO-HalfCheetah configuration
├── experiments/
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Evaluation and Pareto front analysis
│   ├── run_baselines.py          # Baseline comparison script
│   └── reproduce_paper.sh        # Script to reproduce all paper results
├── analysis/
│   ├── plot_pareto.py            # Pareto front visualization
│   ├── plot_ablation.py          # Ablation study plots
│   └── statistical_tests.py      # Wilcoxon tests with Holm-Bonferroni correction
├── tests/
│   ├── test_metrics.py           # Unit tests for interpretability metrics
│   └── test_decision_tree.py     # Unit tests for GE and decision trees
└── data/
    └── README.md                 # Data and results directory
```

## Installation

### Requirements

- Python 3.9 or higher
- MuJoCo 2.3+ (for Hopper and MO-HalfCheetah environments)

### Install dependencies

```bash
cd Interpretable_RL_code
pip install -e .
```

Or install directly from requirements:

```bash
pip install -r requirements.txt
```

### MuJoCo setup

For Hopper and MO-HalfCheetah, MuJoCo must be installed separately. Follow the [official MuJoCo installation guide](https://github.com/openai/mujoco-py#install-mujoco).

## Quick Start

### Training on CartPole (performance vs. interpretability)

```bash
python experiments/train.py --config configs/cartpole.yaml --algorithm nsga3 --seed 42
```

### Training on Deep Sea Treasure (multi-objective benchmark)

```bash
python experiments/train.py --config configs/dst.yaml --algorithm nsga3 --seed 42
```

### Evaluating a trained Pareto front

```bash
python experiments/evaluate.py --results results/cartpole_nsga3_seed42/ --plot
```

### Running all baselines

```bash
python experiments/run_baselines.py --config configs/cartpole.yaml
```

### Reproducing all paper results

```bash
bash experiments/reproduce_paper.sh
```

## Interpretability Metric

The composite interpretability proxy metric is defined as:

$$I(\mathcal{T}) = w_1 M_\text{depth} + w_2 M_\text{rules} + w_3 M_\text{temporal} + w_4 M_\text{causal}$$

with default weights $(w_1, w_2, w_3, w_4) = (0.3, 0.3, 0.2, 0.2)$.

Each component:
- **M_depth**: Normalized inverse tree depth — shallower trees score higher
- **M_rules**: Normalized inverse leaf count — fewer rules score higher
- **M_temporal**: Action consistency along trajectories — lower switching scores higher
- **M_causal**: Fraction of state dimensions tested in tree nodes

For continuous action spaces (Hopper, MO-HalfCheetah), the temporal coherence indicator is replaced by a normalized action-change magnitude:

$$\max\!\left(0,\;1 - \frac{\|a_t - a_{t+1}\|_2}{\sqrt{|\mathcal{A}|}\,a_{\max}}\right)$$

## Configuration

Each environment has its own YAML configuration file. Key parameters:

```yaml
environment:
  id: CartPole-v1
  is_multi_objective: false

ea:
  population_size: 100
  generations: 200
  episodes_per_eval: 5
  discount: 0.99
  max_tree_depth: 5
  tournament_size: 3

ge:
  genotype_length: 100
  crossover_prob: 0.7
  mutation_prob: 0.01
  mutation_indiv_prob: 0.2

nsga3:
  reference_divisions: 12

moead:
  neighborhood_size: 10

interpretability_weights:
  w1: 0.3   # depth weight
  w2: 0.3   # rules weight
  w3: 0.2   # temporal coherence weight
  w4: 0.2   # causal sufficiency weight
```

## Results

Reported results (mean over 10 seeds):

| Environment        | Algorithm | HV     | Sparsity | Perf. Retention |
|--------------------|-----------|--------|----------|-----------------|
| Deep Sea Treasure  | NSGA-III  | 0.847  | 0.031    | —               |
| Deep Sea Treasure  | MOEA/D    | 0.831  | 0.028    | —               |
| CartPole-v1        | NSGA-III  | 0.923  | 0.042    | 98.5%           |
| CartPole-v1        | MOEA/D    | 0.908  | 0.038    | 97.1%           |
| Hopper-v3          | NSGA-III  | 0.782  | 0.057    | 87.3%           |
| MO-HalfCheetah     | NSGA-III  | 0.748  | 0.063    | —               |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chen2025morl_interp,
  title={Multi-Objective Evolutionary Reinforcement Learning for Pareto-Optimal Interpretable Policies},
  author={Chen, Shikun and Liu, Yangguang},
  journal={Applied Soft Computing},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This implementation uses:
- [DEAP](https://github.com/DEAP/deap) for evolutionary algorithm infrastructure
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for RL environments
- [MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium) for multi-objective environments
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for neural network baselines
