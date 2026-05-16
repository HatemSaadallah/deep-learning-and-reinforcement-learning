"""Reproduces the slide-style regret comparison.

Run:
    python plot_regret.py

Produces:
    regret_curves.png       (mean cumulative regret +/- 1 std)
    regret_loglog.png       (log-log to read off the exponent in T)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from env import BernoulliBandit
from agents import (
    FollowTheLeader,
    UniformExploration,
    EpsilonGreedy,
    ExploreThenCommit,
    UCB1,
    ThompsonSamplingBeta,
)
from experiment import average_over_seeds


def main() -> None:
    K = 10
    T = 5_000
    n_seeds = 30

    # Hard-ish instance: best arm at 0.7, others spread around 0.5
    rng = np.random.default_rng(0)
    means = np.concatenate([[0.7], rng.uniform(0.3, 0.6, size=K - 1)])
    print(f"Instance means: {np.round(means, 3)}")
    print(f"Optimal arm: {means.argmax()} (mu* = {means.max():.3f})")
    print(f"Gaps:           {np.round(means.max() - means, 3)}")

    def env_fn(seed):
        return BernoulliBandit(means, seed=seed)

    agents = {
        "Uniform":   lambda seed: UniformExploration(K, seed=seed),
        "FTL":       lambda seed: FollowTheLeader(K, seed=seed),
        "eps=0.1":   lambda seed: EpsilonGreedy(K, epsilon=0.1, seed=seed),
        "ETC":       lambda seed: ExploreThenCommit(K, T=T, seed=seed),
        "UCB1":      lambda seed: UCB1(K, seed=seed),
        "Thompson":  lambda seed: ThompsonSamplingBeta(K, seed=seed),
    }

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fn in agents.items():
        print(f"  running {name} ...")
        results[name] = average_over_seeds(env_fn, fn, T=T, n_seeds=n_seeds)
    print(f"\nFinal regret @ T={T} (mean ± std):")
    for name, (mean, std) in results.items():
        print(f"  {name:10s}  {mean[-1]:8.1f}  ± {std[-1]:6.1f}")

    # --- Linear-scale plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ts = np.arange(1, T + 1)
    for name, (mean, std) in results.items():
        ax.plot(ts, mean, label=name, linewidth=1.6)
        ax.fill_between(ts, mean - std, mean + std, alpha=0.12)
    ax.set_xlabel("round t")
    ax.set_ylabel("cumulative pseudo-regret $R_t$")
    ax.set_title(f"Stochastic bandit: K={K} arms, T={T}, averaged over {n_seeds} seeds")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("regret_curves.png", dpi=140)
    print("wrote regret_curves.png")

    # --- Log-log: read off the exponent ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name, (mean, _) in results.items():
        ax2.loglog(ts, np.maximum(mean, 1e-3), label=name, linewidth=1.6)
    # Reference slopes
    ax2.loglog(ts, 5 * np.sqrt(K * ts), "k--", alpha=0.5, label=r"$\sqrt{KT}$ (LB)")
    ax2.loglog(ts, 2 * ts ** (2 / 3), "k:",  alpha=0.5, label=r"$T^{2/3}$ (ETC UB)")
    ax2.set_xlabel("round t (log)")
    ax2.set_ylabel("cumulative regret (log)")
    ax2.set_title("Regret on log-log axes — slope = exponent in T")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("regret_loglog.png", dpi=140)
    print("wrote regret_loglog.png")


if __name__ == "__main__":
    main()
