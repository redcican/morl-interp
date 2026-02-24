"""
VIPER (Verifiable Reinforcement Learning via Policy Extraction) baseline.

VIPER first trains a neural network policy (PPO) to maximize task performance,
then extracts a decision tree approximation via dataset aggregation (DAgger).
The extracted tree is used as a post-hoc interpretable policy.

This baseline tests whether post-hoc extraction can match the performance
and interpretability achieved by our inherently interpretable approach.

Reference: Bastani et al. (2018), "Verifiable Reinforcement Learning via
Policy Extraction", NeurIPS.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..algorithms.morl_framework import MORLFramework
from ..environments.wrappers import collect_trajectories, make_env
from ..metrics.interpretability import compute_composite_interpretability

logger = logging.getLogger(__name__)


class VIPERBaseline:
    """
    VIPER policy extraction baseline.

    Trains a PPO agent for n_ppo_steps total timesteps, then extracts
    a decision tree approximation using a DAgger-like dataset aggregation
    from the trained neural policy.

    Parameters
    ----------
    framework : MORLFramework for shared configuration and evaluation.
    max_depth : Maximum decision tree depth for CART extraction.
    n_ppo_steps : Total timesteps for PPO training.
    n_dagger_iters : Number of DAgger iterations.
    n_dagger_episodes : Episodes per DAgger iteration.
    """

    def __init__(
        self,
        framework: MORLFramework,
        max_depth: int = 5,
        n_ppo_steps: int = 100_000,
        n_dagger_iters: int = 10,
        n_dagger_episodes: int = 20,
    ) -> None:
        self.framework = framework
        self.config = framework.config
        self.max_depth = max_depth
        self.n_ppo_steps = n_ppo_steps
        self.n_dagger_iters = n_dagger_iters
        self.n_dagger_episodes = n_dagger_episodes

    def run(self) -> Dict:
        """
        Train PPO, extract decision tree via DAgger, evaluate both.

        Returns
        -------
        Dictionary with keys:
            - 'neural_performance': PPO performance before extraction.
            - 'tree_performance': Decision tree performance after extraction.
            - 'interpretability': Composite interpretability score of extracted tree.
            - 'tree_depth': Depth of the extracted tree.
            - 'tree_rules': Number of rules (leaves) of the extracted tree.
        """
        try:
            from stable_baselines3 import PPO
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        except ImportError as e:
            raise ImportError(
                "VIPER requires stable-baselines3 and scikit-learn. "
                "Install with: pip install stable-baselines3 scikit-learn"
            ) from e

        env_cfg = self.config["environment"]
        env_id = env_cfg["id"]

        # --- Step 1: Train PPO teacher ---
        logger.info("VIPER | Training PPO teacher for %d steps...", self.n_ppo_steps)
        import gymnasium as gym
        train_env = gym.make(env_id)
        ppo_model = PPO("MlpPolicy", train_env, verbose=0)
        ppo_model.learn(total_timesteps=self.n_ppo_steps)
        train_env.close()

        # Evaluate neural policy
        neural_perf = self._eval_neural(ppo_model)
        logger.info("  PPO performance: %.3f", neural_perf)

        # --- Step 2: DAgger-style dataset collection ---
        logger.info(
            "VIPER | Collecting DAgger dataset (%d iters × %d episodes)...",
            self.n_dagger_iters,
            self.n_dagger_episodes,
        )
        states_all = []
        actions_all = []
        eval_env = gym.make(env_id)
        is_discrete = self.framework.env_spec.is_discrete

        for it in range(self.n_dagger_iters):
            for _ in range(self.n_dagger_episodes):
                state, _ = eval_env.reset()
                done = False
                while not done:
                    # Query oracle (PPO) for action label
                    action, _ = ppo_model.predict(state, deterministic=True)
                    states_all.append(state.copy())
                    actions_all.append(action)
                    state, _, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
        eval_env.close()

        X = np.array(states_all)
        y = np.array(actions_all)

        # --- Step 3: Fit decision tree ---
        logger.info("VIPER | Fitting CART decision tree (max_depth=%d)...", self.max_depth)
        if is_discrete:
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=42,
            )
        else:
            clf = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=42,
            )
        clf.fit(X, y)

        # Wrap in a callable policy for evaluation
        class SKLearnTreePolicy:
            def __init__(self, clf, is_discrete):
                self.clf = clf
                self._is_discrete = is_discrete
                # Expose attributes needed by interpretability metrics
                self.state_dim = X.shape[1]

            def act(self, state):
                pred = self.clf.predict(state.reshape(1, -1))[0]
                if self._is_discrete:
                    return int(pred)
                return np.asarray(pred, dtype=float)

            # Stub structural properties for interpretability metric
            def get_depth(self):
                return int(clf.get_depth())

            def get_num_rules(self):
                return int(clf.get_n_leaves())

            def get_features_used(self):
                fi = clf.feature_importances_
                return set(int(i) for i in np.where(fi > 0)[0])

        tree_policy = SKLearnTreePolicy(clf, is_discrete)

        # --- Step 4: Evaluate extracted tree ---
        logger.info("VIPER | Evaluating extracted decision tree...")
        tree_perf, action_trajs, _ = collect_trajectories(
            policy=tree_policy,
            config=self.config,
            n_episodes=self.framework.n_episodes,
            discount=self.framework.discount,
        )

        # Compute interpretability metrics
        interp, components = compute_composite_interpretability(
            tree=tree_policy,
            trajectories=action_trajs,
            state_dim=self.framework.env_spec.state_dim,
            weights=self.framework.interp_weights,
            d_min=self.framework.d_min,
            d_max=float(self.framework.d_max),
            r_min=self.framework.r_min,
            r_max=self.framework.r_max,
            is_continuous=not is_discrete,
            action_dim=self.framework.env_spec.n_actions,
            a_max=self.framework.env_spec.a_max,
        )

        result = {
            "neural_performance": neural_perf,
            "tree_performance": float(tree_perf),
            "interpretability": float(interp),
            "tree_depth": clf.get_depth(),
            "tree_rules": clf.get_n_leaves(),
            "components": components,
        }
        logger.info(
            "VIPER | tree_perf=%.3f | interp=%.3f | depth=%d | rules=%d",
            result["tree_performance"],
            result["interpretability"],
            result["tree_depth"],
            result["tree_rules"],
        )
        return result

    def _eval_neural(self, model) -> float:
        """Evaluate a Stable-Baselines3 model for n_episodes."""
        import gymnasium as gym
        env = gym.make(self.config["environment"]["id"])
        total = 0.0
        for _ in range(self.framework.n_episodes):
            state, _ = env.reset()
            ep_reward = 0.0
            done = False
            t = 0
            while not done:
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += (self.framework.discount ** t) * float(reward)
                done = terminated or truncated
                t += 1
            total += ep_reward
        env.close()
        return total / self.framework.n_episodes
