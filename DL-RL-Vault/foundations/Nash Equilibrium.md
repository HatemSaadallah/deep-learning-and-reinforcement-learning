# Nash Equilibrium

**Tags:** #foundational #game-theory #definition

A strategy profile where **no player can unilaterally improve** their utility by deviating.

## Definition

In an $n$-player game with strategy sets $\mathcal{X}_i$ and utility functions $u_i: \prod_i \mathcal{X}_i \to \mathbb{R}$, a profile $(x_1^*, \dots, x_n^*)$ is a **Nash Equilibrium (NE)** if:
$$u_i(x_i^*, x_{-i}^*) \;\geq\; u_i(\hat x_i, x_{-i}^*) \qquad \forall i,\; \forall \hat x_i \in \mathcal{X}_i.$$

i.e. each $x_i^*$ is a **best response** to the others' choices.

## $\epsilon$-approximate NE

The profile is an **$\epsilon$-NE** if no player can improve by more than $\epsilon$:
$$\max_{\hat x_i} u_i(\hat x_i, x_{-i}) - u_i(x_i, x_{-i}) \;\leq\; \epsilon \quad \forall i.$$

This is what no-regret learning dynamics converge to — see [[Regret and Equilibria]].

## Two-player zero-sum case

Strategy sets $\mathcal{X}, \mathcal{Y}$, payoff matrix $A$. P1 maximizes $x^\top A y$, P2 minimizes. NE coincides with the saddle point of the bilinear form:
$$\max_x \min_y x^\top A y \;=\; \min_y \max_x x^\top A y \;=\; v^* \qquad\text{(von Neumann minimax)}.$$

The NE is a solution to the **bilinear saddle-point problem**
$$\max_{x \in \mathcal{X}} \min_{y \in \mathcal{Y}} x^\top A y.$$

## Saddle-point gap (duality gap)

Quantifies how far $(x, y)$ is from being a NE in a zero-sum game:
$$\gamma(x, y) \;:=\; \Big(\max_{\hat x} \hat x^\top A y - x^\top A y\Big) \;+\; \Big(x^\top A y - \min_{\hat y} x^\top A \hat y\Big) \;=\; \max_{\hat x} \hat x^\top A y - \min_{\hat y} x^\top A \hat y.$$

- $\gamma(x, y) \geq 0$, with equality iff $(x, y)$ is a NE.
- $\gamma(x, y) \leq \epsilon$ iff $(x, y)$ is an $\epsilon$-NE.
- The metric used to measure convergence of [[Regret and Equilibria|no-regret dynamics]] to equilibrium.

## Equilibrium hierarchy (where Nash sits)

| Solution concept | Condition | Computational complexity |
|---|---|---|
| **Dominant strategy** | best response regardless of others | trivial when exists; rarely exists |
| **Nash** | best response to opponents' actual play | PPAD-hard in general; **polytime in zero-sum** |
| **Correlated equilibrium** | best response under a correlation device | polytime via LP |
| **Coarse correlated equilibrium** | no swap-deviation profitable | polytime; what no-regret dynamics converge to in general-sum |

In general-sum games, no-regret learning **does not converge to NE** — only to CCE. The zero-sum case is special because CCE marginals = NE.

## Why it matters for the course

- **Two-player zero-sum games**: no-regret dynamics converge to NE at rate $O(1/\sqrt{T})$, or $O(1/T)$ with [[Optimistic FTRL]]. See [[Regret and Equilibria]].
- **RL connection:** multi-agent RL convergence often analyzed through this lens; self-play poker / game-AI systems (CFR, Pluribus) use no-regret minimization at scale.

## See also

- [[Regret and Equilibria]] — the regret → Nash convergence proof.
- [[Optimistic FTRL]] — gets $O(1/T)$ rates in games.
- [[Hedge - Multiplicative Weights]] — the canonical algorithm for self-play.
