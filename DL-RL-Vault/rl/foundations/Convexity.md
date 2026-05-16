# Convexity

**Tags:** #foundational #convex-analysis #definition

The geometric / analytic hypothesis under which [[Online Convex Optimization|OCO]] and most of online learning becomes tractable.

## Convex set

A set $\mathcal{C} \subseteq \mathbb{R}^d$ is **convex** if for any $x, y \in \mathcal{C}$ and $\lambda \in [0, 1]$:
$$\lambda x + (1-\lambda) y \in \mathcal{C}.$$

i.e. every line segment between two points of $\mathcal{C}$ lies in $\mathcal{C}$.

**Examples:** $\mathbb{R}^d$, halfspaces, hyperplanes, balls $B(x_0, R) = \{x : \|x - x_0\| \leq R\}$, simplex $\Delta_N = \{p \in \mathbb{R}^N_{\geq 0} : \sum p_i = 1\}$, intersection of convex sets, positive semidefinite cone.

## Convex function

A function $f: \mathcal{C} \to \mathbb{R}$ (with $\mathcal{C}$ convex) is **convex** if for all $x, y \in \mathcal{C}$, $\lambda \in [0, 1]$:
$$f(\lambda x + (1-\lambda) y) \;\leq\; \lambda f(x) + (1-\lambda) f(y).$$

The function value at a midpoint never exceeds the average of the values.

### Equivalent characterizations

For differentiable $f$ on a convex domain:

1. **First-order (subgradient inequality):** for all $x, y$,
$$f(y) \;\geq\; f(x) + \langle \nabla f(x),\, y - x\rangle.$$
(Tangent plane is a global underestimate.) Holds with $g \in \partial f(x)$ for non-differentiable convex $f$.

2. **Second-order:** if $f \in C^2$, then $\nabla^2 f(x) \succeq 0$ everywhere.

3. **Monotone gradient:** $\langle \nabla f(x) - \nabla f(y),\, x - y\rangle \geq 0$.

## Strict vs. strong vs. exp-concave (regularity hierarchy)

Stronger curvature gives faster algorithms ($\sqrt{T} \to \log T$ regret).

| Property | Definition | Implies |
|---|---|---|
| **Convex** | (above) | $O(\sqrt{T})$ regret in OCO |
| **Strictly convex** | strict inequality for $x \neq y$, $\lambda \in (0, 1)$ | unique minimizer (when one exists) |
| **$\mu$-strongly convex** w.r.t. $\|\cdot\|$ | $f(y) \geq f(x) + \langle\nabla f(x), y-x\rangle + \tfrac{\mu}{2}\|y - x\|^2$ | $O((1/\mu) \log T)$ regret |
| **$\alpha$-[[Exp-Concavity\|exp-concave]]** | $\exp(-\alpha f)$ is concave | $O((d/\alpha) \log T)$ regret |
| Smooth ($L$-Lipschitz gradient) | $\nabla^2 f \preceq L I$ | accelerated rates available |

**Strongly convex ⟹ exp-concave ⟹ strictly convex ⟹ convex** (each direction strict).

Strong convexity is equivalent to a quadratic lower bound on the [[Bregman Divergence|Bregman divergence]]:
$$D_f(y, x) \geq \frac{\mu}{2}\|y - x\|^2.$$

## Why this matters for online learning

- The **subgradient inequality** linearizes regret: $\ell_t(\theta_t) - \ell_t(\theta^*) \leq \langle g_t, \theta_t - \theta^*\rangle$. So a bound on $\sum \langle g_t, \theta_t - \theta^*\rangle$ becomes a regret bound.
- **Convex projections** $\Pi_\mathcal{C}$ contract distances to feasible points → key step in the [[OGD Regret Bound|OGD proof]].
- **Strong convexity** (or exp-concavity) of *losses* gives faster regret. Strong convexity of the *regularizer* gives [[FTRL Stability and Regret|stability of FTRL iterates]].

## Common convex loss functions in ML

| Loss | Formula | Used in |
|---|---|---|
| Squared error | $(y - \langle \theta, x\rangle)^2$ | Linear / [[Online Ridge Regression\|ridge]] regression. Exp-concave on bounded data. |
| Hinge | $\max(0, 1 - y \langle \theta, x\rangle)$ | SVM. Convex, non-smooth. |
| Logistic | $\log(1 + \exp(-y \langle \theta, x\rangle))$ | [[Online Logistic Regression]]. Convex, smooth, exp-concave on bounded domains. |
| Cross-entropy | $-\sum_i y_i \log p_i$ | Classification. Convex in logits up to softmax. |
| Negative log-likelihood (exp. fam.) | $-\log p_\theta(y)$ | Generalized linear models. Convex, exp-concave. |

## See also

- [[Bregman Divergence]] — the convex-analysis tool that powers MD and FTRL.
- [[Online Convex Optimization]] — the framework where convexity is the standing assumption.
- [[Exp-Concavity]] — stronger curvature giving log-$T$ regret.
