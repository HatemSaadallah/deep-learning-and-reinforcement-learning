"""Bandit algorithms. Each agent exposes:
    select() -> arm index
    update(a, r) -> None
    name -> str
"""
from __future__ import annotations
import numpy as np


class Agent:
    def __init__(self, K: int, seed: int | None = None):
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.N = np.zeros(K, dtype=int)        # N_a(t): pull counts
        self.S = np.zeros(K, dtype=float)      # cumulative reward per arm
        self.t = 0

    @property
    def mu_hat(self) -> np.ndarray:
        """Empirical means; arms with N=0 get +inf so they're tried first."""
        out = np.full(self.K, np.inf)
        seen = self.N > 0
        out[seen] = self.S[seen] / self.N[seen]
        return out

    def update(self, a: int, r: float) -> None:
        self.N[a] += 1
        self.S[a] += r
        self.t += 1

    def select(self) -> int:
        raise NotImplementedError


class FollowTheLeader(Agent):
    """Pure greedy. Lecture slide 12: suffers Omega(T) regret."""
    name = "FTL (greedy)"

    def select(self) -> int:
        return int(np.argmax(self.mu_hat))


class UniformExploration(Agent):
    """Random arm every round. Lecture slide 11: linear regret = T * mean(Delta)."""
    name = "Uniform"

    def select(self) -> int:
        return int(self.rng.integers(self.K))


class EpsilonGreedy(Agent):
    """With prob epsilon explore uniformly, else exploit."""

    def __init__(self, K: int, epsilon: float = 0.1, seed: int | None = None):
        super().__init__(K, seed)
        self.epsilon = epsilon
        self.name = f"eps-greedy (eps={epsilon})"

    def select(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.K))
        return int(np.argmax(self.mu_hat))


class ExploreThenCommit(Agent):
    """ETC with T_0 = (T/K)^(2/3) * (log T)^(1/3). Lecture slides 13-17."""
    def __init__(self, K: int, T: int, seed: int | None = None):
        super().__init__(K, seed)
        self.T = T
        self.T0 = max(1, int(np.ceil((T / K) ** (2 / 3) * np.log(max(T, 2)) ** (1 / 3))))
        self.committed_arm: int | None = None
        self.name = f"ETC (T0={self.T0})"

    def select(self) -> int:
        if self.t < self.K * self.T0:
            # Round-robin exploration
            return self.t % self.K
        if self.committed_arm is None:
            self.committed_arm = int(np.argmax(self.mu_hat))
        return self.committed_arm


class UCB1(Agent):
    """Auer et al. 2002. a_t = argmax mu_hat(a) + sqrt(2 log t / N_a)."""
    name = "UCB1"

    def select(self) -> int:
        unseen = np.where(self.N == 0)[0]
        if len(unseen) > 0:
            return int(unseen[0])
        bonus = np.sqrt(2.0 * np.log(max(self.t, 1)) / self.N)
        return int(np.argmax(self.S / self.N + bonus))


class ThompsonSamplingBeta(Agent):
    """Bernoulli Thompson Sampling: maintain Beta(alpha, beta) per arm."""
    name = "Thompson Sampling"

    def __init__(self, K: int, seed: int | None = None):
        super().__init__(K, seed)
        self.alpha = np.ones(K)  # successes + 1
        self.beta = np.ones(K)   # failures + 1

    def select(self) -> int:
        theta = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(theta))

    def update(self, a: int, r: float) -> None:
        super().update(a, r)
        # r is 0/1 for Bernoulli; clip just in case it's a float close to those
        self.alpha[a] += r
        self.beta[a] += (1.0 - r)
