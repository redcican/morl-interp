"""
Environment factory and trajectory collection utilities.

Supports both single-objective environments (gymnasium) and multi-objective
environments (mo-gymnasium). For single-objective environments, the standard
reward signal is used as the performance objective. For multi-objective
environments, the full reward vector is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Environment specification
# ---------------------------------------------------------------------------


@dataclass
class EnvSpec:
    """Metadata describing an environment used in this study."""

    env_id: str
    is_multi_objective: bool
    state_dim: int
    is_discrete: bool
    n_actions: int                     # number of actions (discrete) or dims (continuous)
    action_low: Optional[np.ndarray]   # None for discrete
    action_high: Optional[np.ndarray]  # None for discrete
    a_max: float                        # max per-dim action magnitude
    description: str = ""

    # Known true Pareto front (only DST has one analytically)
    true_pareto_front: Optional[np.ndarray] = field(default=None, repr=False)


# Ground-truth Pareto front for Deep Sea Treasure (Vamplew et al., 2011)
# Columns: [treasure_value, time_penalty]  (both maximized after sign flip)
DST_TRUE_PARETO = np.array([
    [1.0,  -1.0],
    [2.0,  -3.0],
    [3.0,  -5.0],
    [5.0,  -7.0],
    [8.0, -8.0],
    [16.0, -9.0],
    [24.0, -13.0],
    [50.0, -14.0],
    [74.0, -17.0],
    [124.0, -19.0],
], dtype=float)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env(config: Dict[str, Any], render: bool = False):
    """
    Create and return an environment from the configuration dictionary.

    Parameters
    ----------
    config : Dictionary with at least config['environment']['id'] and
             config['environment']['is_multi_objective'].
    render : Whether to enable rendering (for visualization only).

    Returns
    -------
    A gymnasium or mo-gymnasium environment instance.
    """
    env_cfg = config["environment"]
    env_id = env_cfg["id"]
    is_mo = env_cfg.get("is_multi_objective", False)
    render_mode = "human" if render else None

    if is_mo:
        try:
            import mo_gymnasium as mo_gym
            env = mo_gym.make(env_id, render_mode=render_mode)
        except ImportError as e:
            raise ImportError(
                "mo-gymnasium is required for multi-objective environments. "
                "Install it with: pip install mo-gymnasium"
            ) from e
    else:
        import gymnasium as gym
        env = gym.make(env_id, render_mode=render_mode)

    return env


def get_env_spec(config: Dict[str, Any]) -> EnvSpec:
    """
    Inspect an environment and return its EnvSpec.

    The environment is created, queried, and immediately closed.
    """
    env = make_env(config)
    env_cfg = config["environment"]
    is_mo = env_cfg.get("is_multi_objective", False)

    obs_space = env.observation_space
    act_space = env.action_space
    state_dim = int(obs_space.shape[0])

    if hasattr(act_space, "n"):
        is_discrete = True
        n_actions = int(act_space.n)
        action_low = None
        action_high = None
        a_max = 1.0
    else:
        is_discrete = False
        n_actions = int(act_space.shape[0])
        action_low = act_space.low.astype(float)
        action_high = act_space.high.astype(float)
        a_max = float(act_space.high.max())

    env.close()

    true_pareto = None
    if env_cfg["id"].lower().startswith("deep") or "dst" in env_cfg["id"].lower():
        true_pareto = DST_TRUE_PARETO

    return EnvSpec(
        env_id=env_cfg["id"],
        is_multi_objective=is_mo,
        state_dim=state_dim,
        is_discrete=is_discrete,
        n_actions=n_actions,
        action_low=action_low,
        action_high=action_high,
        a_max=a_max,
        true_pareto_front=true_pareto,
    )


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------


def collect_trajectories(
    policy,
    config: Dict[str, Any],
    n_episodes: int = 5,
    discount: float = 0.99,
    seed: Optional[int] = None,
) -> Tuple[float, List[List], Optional[np.ndarray]]:
    """
    Run a policy in the environment for n_episodes and collect action
    trajectories for interpretability metric computation.

    Parameters
    ----------
    policy     : A DecisionTreePolicy with an act(state) method.
    config     : Experiment configuration dictionary.
    n_episodes : Number of evaluation episodes.
    discount   : Discount factor for cumulative reward computation.
    seed       : Optional random seed for reproducibility.

    Returns
    -------
    avg_perf       : Average discounted cumulative return (scalar for
                     single-objective; sum of objectives for multi-objective).
    action_trajs   : List of action sequences [[a_0, a_1, ...], ...].
    vec_reward_avg : Average reward vector (None for single-objective envs).
    """
    env = make_env(config)
    is_mo = config["environment"].get("is_multi_objective", False)

    total_scalar = 0.0
    total_vec: Optional[np.ndarray] = None
    action_trajs: List[List] = []

    for ep in range(n_episodes):
        reset_kwargs = {"seed": seed + ep} if seed is not None else {}
        state, _ = env.reset(**reset_kwargs)

        episode_actions: List = []
        episode_scalar = 0.0
        episode_vec: Optional[np.ndarray] = None
        done = False
        t = 0

        while not done:
            action = policy.act(np.asarray(state, dtype=float))
            obs, reward, terminated, truncated, _ = env.step(action)

            # Accumulate reward
            if is_mo:
                r = np.asarray(reward, dtype=float)
                if episode_vec is None:
                    episode_vec = np.zeros_like(r)
                episode_vec += (discount ** t) * r
                episode_scalar += (discount ** t) * float(r.sum())
            else:
                episode_scalar += (discount ** t) * float(reward)

            episode_actions.append(action)
            state = obs
            done = terminated or truncated
            t += 1

        total_scalar += episode_scalar
        action_trajs.append(episode_actions)

        if is_mo and episode_vec is not None:
            if total_vec is None:
                total_vec = np.zeros_like(episode_vec)
            total_vec += episode_vec

    env.close()

    avg_perf = total_scalar / n_episodes
    vec_reward_avg = (total_vec / n_episodes) if total_vec is not None else None

    return avg_perf, action_trajs, vec_reward_avg
