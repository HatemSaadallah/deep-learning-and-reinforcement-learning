# Online Ridge Regression (Recursive Least Squares)

**Tags:** #online-learning #algorithm

The online version of ridge / least-squares regression. Achieves $O(d \log T)$ regret thanks to [[Exp-Concavity|exp-concavity]] of the squared loss — exponentially better than what generic OGD would give on a non-strongly-convex problem.

## Setting

[[Online Convex Optimization|OCO]] specialized to squared loss:
- At round $t$: predict $\hat y_t = \langle \theta_t, x_t\rangle$ for revealed feature $x_t \in \mathbb{R}^d$.
- Observe true label $y_t \in \mathbb{R}$, suffer loss $\ell_t(\theta) = (y_t - \langle \theta, x_t\rangle)^2$.

Goal: low [[Regret]] vs. the best fixed linear predictor.

## Why the offline closed form doesn't directly transfer

Batch least squares:
$$\hat\theta_n = (X_n^\top X_n)^{-1} X_n^\top y_n.$$

In an online setting, computing $\hat\theta_n$ at each round requires:
- Storing the **growing** matrix $X_n \in \mathbb{R}^{n \times d}$ and vector $y_n \in \mathbb{R}^n$.
- Re-inverting a $d \times d$ matrix every round.

Both are wasteful — we want a constant-per-round update.

## Recursive update via Sherman-Morrison

Define
$$A_n := \sum_{t=1}^n x_t x_t^\top + \lambda I, \qquad b_n := \sum_{t=1}^n y_t x_t.$$
Then $\hat\theta_n = A_n^{-1} b_n$, and incrementally:
$$A_{n+1} = A_n + x_{n+1} x_{n+1}^\top, \qquad b_{n+1} = b_n + y_{n+1} x_{n+1}.$$

**Sherman-Morrison formula:** for invertible $A$ and vectors $u, v$,
$$(A + uv^\top)^{-1} \;=\; A^{-1} - \frac{A^{-1} u v^\top A^{-1}}{1 + v^\top A^{-1} u}.$$

Applying with $u = v = x_{n+1}$:
$$A_{n+1}^{-1} \;=\; A_n^{-1} - \frac{A_n^{-1} x_{n+1} x_{n+1}^\top A_n^{-1}}{1 + x_{n+1}^\top A_n^{-1} x_{n+1}}.$$

**Per-round complexity:** $O(d^2)$ time, $O(d^2)$ memory (one $d \times d$ matrix + one $d$-vector). Independent of $n$!

## Regret guarantee

For bounded features $\|x_t\| \leq X$ and bounded targets $|y_t| \leq Y$, online ridge regression achieves
$$R_T \;=\; O\!\left(d \log T\right)$$
(constants depending on $X, Y$, regularization $\lambda$).

This is **exponentially better** than the $O(\sqrt{T})$ that generic OGD gives on a Lipschitz convex problem.

**Why this works:** the squared loss is $\alpha$-[[Exp-Concavity|exp-concave]] on bounded data. Exp-concave losses admit second-order online algorithms (Online Newton Step / online ridge regression) with $O((d/\alpha)\log T)$ regret. See Section 4.2 of Hazan (2022).

## Connection to [[LinUCB]]

The matrix $A_n = \lambda I + \sum_s x_s x_s^\top$ that drives online ridge regression is **exactly** the matrix LinUCB uses for the confidence ellipsoid. LinUCB = online ridge regression + UCB-style exploration bonus.

## Intuition / what to remember

- **Sherman-Morrison is the right tool** whenever you incrementally update an inverse with rank-1 modifications.
- **Squared loss is special:** its quadratic curvature makes online learning much faster than for Lipschitz losses. This generalizes to all [[Exp-Concavity|exp-concave]] losses.
- **The closed-form trick fails for non-quadratic losses.** This is why [[Online Logistic Regression]] is harder (no closed form even offline).

## References

- Vovk, *Competitive on-line statistics*, ISR 2001 — first formal analysis.
- Azoury & Warmuth — exponential weighted update for ridge.
- Hazan 2022, §4.2 — exp-concavity and Online Newton Step.

## See also

- [[Online Convex Optimization]] — the framework.
- [[Exp-Concavity]] — why log $T$ regret is achievable.
- [[LinUCB]] — exploits the same matrix algebra.
- [[Online Logistic Regression]] — the harder sibling without a closed form.
