"""Run a single bandit experiment, return the cumulative pseudo-regret curve."""
from __future__ import annotations
import numpy as np


def run_episode(env, agent, T: int) -> np.ndarray:
    """Returns array of length T: cumulative pseudo-regret at each round.

    Pseudo-regret at round t:  t * mu_star - sum_{s<=t} mu_{a_s}
    (uses the true mean of the played arm, not the realized reward)
    """
    regret = np.empty(T)
    cum = 0.0
    for t in range(T):
        a = agent.select()
        r = env.pull(a)
        agent.update(a, r)
        cum += env.gaps[a]  # Delta_{a_t}, instantaneous pseudo-regret
        regret[t] = cum
    return regret


def average_over_seeds(env_fn, agent_fn, T: int, n_seeds: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Average cumulative regret over n_seeds independent runs.
    Returns (mean_regret, std_regret), each shape (T,).
    """
    runs = np.empty((n_seeds, T))
    for s in range(n_seeds):
        env = env_fn(seed=s)
        agent = agent_fn(seed=1000 + s)
        runs[s] = run_episode(env, agent, T)
    return runs.mean(axis=0), runs.std(axis=0)
