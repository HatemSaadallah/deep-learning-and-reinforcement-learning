# Regret

**Tags:** #online-learning #foundational #definition

The performance measure for [[Online Learning]] algorithms.

## Definition

After $T$ rounds:
$$R_T = \sum_{t=1}^T f_t(x_t) \;-\; \min_{x \in \mathcal{K}} \sum_{t=1}^T f_t(x).$$

Compares your cumulative loss to the **best fixed action in hindsight**.

## Sublinear regret = learning

- $R_T = o(T)$ means average regret $\to 0$ → you're competitive with the best fixed action.
- $R_T = O(\sqrt{T})$ is typical for general convex losses (e.g. [[Follow the Regularized Leader]]).
- $R_T = O(\log T)$ achievable for strongly convex losses (e.g. [[Follow the Leader]] with quadratic losses).
- $R_T = \Theta(T)$ means no learning.

## Variants

- **Static regret:** vs. best fixed action (above).
- **Dynamic regret:** vs. best sequence of actions — harder.
- **Swap regret:** stronger notion, equilibrium in [[Self-Play and Game-Theoretic RL]].
