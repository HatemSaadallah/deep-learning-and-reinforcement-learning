"""Bandit environments. Notation matches the lecture slides:
    K       number of arms
    mu_a    true mean of arm a
    mu_star = max_a mu_a
    Delta_a = mu_star - mu_a   (sub-optimality gap)
"""
from __future__ import annotations
import numpy as np


class BernoulliBandit:
    """K-armed Bernoulli bandit: r_t ~ Bernoulli(mu[a_t])."""

    def __init__(self, means: np.ndarray, seed: int | None = None):
        self.means = np.asarray(means, dtype=float)
        assert ((0.0 <= self.means) & (self.means <= 1.0)).all()
        self.K = len(self.means)
        self.mu_star = float(self.means.max())
        self.optimal_arm = int(self.means.argmax())
        self.gaps = self.mu_star - self.means  # Delta_a
        self.rng = np.random.default_rng(seed)

    def pull(self, a: int) -> float:
        return float(self.rng.random() < self.means[a])


class GaussianBandit:
    """K-armed Gaussian bandit with shared variance sigma^2."""

    def __init__(self, means: np.ndarray, sigma: float = 1.0, seed: int | None = None):
        self.means = np.asarray(means, dtype=float)
        self.sigma = sigma
        self.K = len(self.means)
        self.mu_star = float(self.means.max())
        self.optimal_arm = int(self.means.argmax())
        self.gaps = self.mu_star - self.means
        self.rng = np.random.default_rng(seed)

    def pull(self, a: int) -> float:
        return float(self.rng.normal(self.means[a], self.sigma))
