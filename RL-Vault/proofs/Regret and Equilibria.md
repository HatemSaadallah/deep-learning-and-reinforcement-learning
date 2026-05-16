# Proof — No-Regret Dynamics Converge to Nash in Zero-Sum Games

**Tags:** #proof #game-theory #online-learning
**Topic:** Connection between [[Regret]] and equilibria in two-player zero-sum games.

## Statement

Consider a two-player zero-sum matrix game with payoff matrix $A \in \mathbb{R}^{n \times m}$. P1 picks $x \in \Delta_n$ to maximize $x^\top A y$, P2 picks $y \in \Delta_m$ to minimize it. Let $v^* = \max_x \min_y x^\top A y = \min_y \max_x x^\top A y$ (von Neumann minimax theorem).

Suppose both players run **no-regret algorithms** with regret $R_T^1, R_T^2 = o(T)$ over $T$ rounds. Let $\bar x_T = \tfrac{1}{T}\sum_t x_t$, $\bar y_T = \tfrac{1}{T}\sum_t y_t$.

**Theorem:** $(\bar x_T, \bar y_T)$ is a $\bigl((R_T^1 + R_T^2)/T\bigr)$-Nash equilibrium:
$$\boxed{\;\max_x x^\top A \bar y_T \;-\; \min_y \bar x_T^\top A y \;\leq\; \frac{R_T^1 + R_T^2}{T} \;\xrightarrow{T \to \infty}\; 0.\;}$$

In particular, $|\bar x_T^\top A \bar y_T - v^*| \leq (R_T^1 + R_T^2)/T \to 0$.

## Setup

- Each round $t$: P1 plays $x_t$, P2 plays $y_t$, simultaneously.
- P1's loss sequence (it wants to maximize, so its losses are negative payoffs): $\ell^1_t(x) = -x^\top A y_t$. Linear in $x$.
- P2's loss sequence: $\ell^2_t(y) = x_t^\top A y$. Linear in $y$.
- Regret guarantees:
  - **P1:** $\sum_t \ell^1_t(x_t) - \min_x \sum_t \ell^1_t(x) \leq R_T^1$, i.e. $\max_x \sum_t x^\top A y_t - \sum_t x_t^\top A y_t \leq R_T^1$.
  - **P2:** $\sum_t \ell^2_t(y_t) - \min_y \sum_t \ell^2_t(y) \leq R_T^2$, i.e. $\sum_t x_t^\top A y_t - \min_y \sum_t x_t^\top A y \leq R_T^2$.

## Proof

**Step 1 — Rewrite regret guarantees with averages.**

P1's guarantee, divided by $T$:
$$\max_x x^\top A \bar y_T \;-\; \frac{1}{T}\sum_t x_t^\top A y_t \;\leq\; \frac{R_T^1}{T}. \tag{1}$$

P2's guarantee, divided by $T$:
$$\frac{1}{T}\sum_t x_t^\top A y_t \;-\; \min_y \bar x_T^\top A y \;\leq\; \frac{R_T^2}{T}. \tag{2}$$

**Step 2 — Sum (1) and (2).**

The middle terms cancel:
$$\max_x x^\top A \bar y_T \;-\; \min_y \bar x_T^\top A y \;\leq\; \frac{R_T^1 + R_T^2}{T}. \tag{$\star$}$$

**Step 3 — Interpret as duality gap.**

By the minimax theorem:
$$\max_x x^\top A \bar y_T \;\geq\; \min_{y'} \max_x x^\top A y' \;=\; v^*,$$
$$\min_y \bar x_T^\top A y \;\leq\; \max_{x'} \min_y x'^\top A y \;=\; v^*.$$

So both quantities in $(\star)$ sandwich $v^*$:
$$0 \;\leq\; \max_x x^\top A \bar y_T - v^* \;\leq\; (R_T^1 + R_T^2)/T,$$
$$0 \;\leq\; v^* - \min_y \bar x_T^\top A y \;\leq\; (R_T^1 + R_T^2)/T.$$

The first inequality says $\bar y_T$ is an $\epsilon$-optimal P2 strategy (no $x$ exploits it by more than $\epsilon$). The second says symmetrically for $\bar x_T$. Together, $(\bar x_T, \bar y_T)$ is an $\epsilon$-Nash with $\epsilon = (R_T^1 + R_T^2)/T$. $\square$

## Rates with concrete algorithms

If both players run [[Hedge - Multiplicative Weights|Hedge]] with appropriate step size, $R_T = O(\sqrt{T \log N})$, giving an $O(\sqrt{\log N / T})$-Nash after $T$ rounds. To reach $\epsilon$-Nash, need
$$T = O\!\left(\frac{\log N}{\epsilon^2}\right).$$

This is the **standard "no-regret learning solves zero-sum games"** result.

## Stronger regret notions → stronger equilibria

| Regret notion | Equilibrium concept |
|---|---|
| External regret (above) | Coarse correlated equilibrium / Nash in zero-sum games |
| **Swap regret** | Correlated equilibrium |
| Internal regret | Correlated equilibrium |

For general-sum games, no-regret learning converges to the *coarse correlated equilibrium*, not Nash. Nash for general-sum games is PPAD-hard.

## Intuition / what to remember

- **No-regret = at-least-as-good-as-best-fixed-action.** When two such players meet in a zero-sum game, the only stable point is Nash.
- **Time averaging is essential.** Individual iterates $(x_t, y_t)$ generally do NOT converge to Nash — they cycle. The *averages* converge.
- **Two regret guarantees + minimax = $\epsilon$-Nash.** The proof is essentially algebraic: add the two inequalities, the cross-term cancels.
- **Practical takeaway:** to solve a zero-sum game (poker, security games, robust optimization), run any no-regret algorithm against itself and average the iterates. This is the recipe behind CFR (Counterfactual Regret Minimization) for poker.

## See also

- [[Regret]] — the quantity being bounded.
- [[Hedge - Multiplicative Weights]] — the canonical no-regret algorithm.
- [[Follow the Regularized Leader]] / [[Mirror Descent Analysis]] — algorithm families with sublinear regret.
