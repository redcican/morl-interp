#!/usr/bin/env bash
# =============================================================================
# Reproduce all main results from the paper.
#
# This script runs NSGA-III and MOEA/D on all four environments across 10
# random seeds, then runs baselines and generates figures.
#
# Estimated runtime: 24–48 hours on a single CPU.
# For faster runs, reduce --generations in each config or use fewer seeds.
#
# Usage:
#   bash experiments/reproduce_paper.sh
#   bash experiments/reproduce_paper.sh --seeds 3    # quick test with 3 seeds
# =============================================================================

set -euo pipefail

# Parse optional --seeds argument
N_SEEDS=10
if [[ "${1:-}" == "--seeds" && -n "${2:-}" ]]; then
    N_SEEDS="$2"
fi

echo "=================================================="
echo "  Reproducing paper results (${N_SEEDS} seeds)"
echo "=================================================="

CONFIGS=(
    "configs/cartpole.yaml"
    "configs/dst.yaml"
    "configs/hopper.yaml"
    "configs/mo_halfcheetah.yaml"
)

ALGORITHMS=("nsga3" "moead")

# --- Main experiments ---
for CONFIG in "${CONFIGS[@]}"; do
    for ALG in "${ALGORITHMS[@]}"; do
        for SEED in $(seq 0 $((N_SEEDS - 1))); do
            echo ">>> $CONFIG | $ALG | seed=$SEED"
            python experiments/train.py \
                --config "$CONFIG" \
                --algorithm "$ALG" \
                --seed "$SEED"
        done
    done
done

# --- Baselines ---
echo ""
echo "--- Running baselines ---"
for CONFIG in "${CONFIGS[@]}"; do
    for SEED in $(seq 0 $((N_SEEDS - 1))); do
        echo ">>> baselines | $CONFIG | seed=$SEED"
        python experiments/run_baselines.py \
            --config "$CONFIG" \
            --seed "$SEED"
    done
done

# --- Analysis ---
echo ""
echo "--- Generating figures ---"
python analysis/plot_pareto.py --results-dir results/
python analysis/plot_ablation.py --results-dir results/

echo ""
echo "--- Statistical tests ---"
python analysis/statistical_tests.py --results-dir results/

echo ""
echo "All done. Results are in results/. Figures are in results/figures/."
