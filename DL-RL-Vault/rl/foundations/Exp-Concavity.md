# Exp-Concavity

**Tags:** #foundational #convex-analysis #definition

A regularity condition on convex losses that is **weaker than strong convexity** but **stronger than convexity** — and just enough to unlock $O(d \log T)$ regret in [[Online Convex Optimization|OCO]].

## Definition

A function $f: \mathcal{C} \to \mathbb{R}$ is **$\alpha$-exp-concave** ($\alpha > 0$) if the function $\exp(-\alpha f)$ is **concave** on $\mathcal{C}$.

**Equivalent characterization** (for twice-differentiable $f$):
$$\nabla^2 f(\theta) \;\succeq\; \alpha\, \nabla f(\theta)\, \nabla f(\theta)^\top \qquad \forall \theta \in \mathcal{C}.$$

In words: **the Hessian must dominate the outer product of the gradient.** The function curves at least as fast as its gradient grows.

## Hierarchy

$$\text{strongly convex}\;\Longrightarrow\; \text{exp-concave}\;\Longrightarrow\; \text{strictly convex}\;\Longrightarrow\; \text{convex}.$$

(All directions strict in general.)

| Property | Best OCO regret |
|---|---|
| Convex + Lipschitz | $\Theta(\sqrt{T})$ |
| **$\alpha$-exp-concave** | $\Theta((d/\alpha) \log T)$ |
| $\mu$-strongly convex | $\Theta((1/\mu) \log T)$ |

So exp-concavity gives logarithmic regret in $T$, with the price of a dimension factor $d$.

## Why "exp-concave" unlocks log-$T$ regret

The condition $\nabla^2 f \succeq \alpha \nabla f \nabla f^\top$ means a **Newton-style** update with the gradient outer-product matrix
$$A_t = \lambda I + \sum_{s \leq t} \nabla \ell_s(\theta_s) \nabla \ell_s(\theta_s)^\top$$
captures the true second-order geometry of the loss accurately enough to bound regret by $\sum \log(\det A_t / \det A_{t-1}) = O(d \log T)$.

The algorithm is **Online Newton Step (ONS)**:
$$\theta_{t+1} = \Pi^{A_t}_\mathcal{C}\!\left(\theta_t - \frac{1}{\alpha} A_t^{-1} g_t\right)$$
where $\Pi^A_\mathcal{C}$ is the projection in the $\|\cdot\|_A$ norm.

For squared loss specifically, ONS reduces to **[[Online Ridge Regression|online ridge regression]]** (recursive least squares).

## Key examples

| Loss | Domain | Exp-concave? |
|---|---|---|
| Squared loss $(y - \langle \theta, x\rangle)^2$ | $\|\theta\|, \|x\|, |y|$ bounded | **Yes**, $\alpha \propto 1/D^2$ where $D$ bounds everything |
| Logistic loss $\ln(1 + e^{-y\langle \theta, x\rangle})$ | $\|\theta\|, \|x\|$ bounded | **Yes**, but the $\alpha$ degrades exponentially with domain radius |
| Hinge loss $\max(0, 1 - y\langle\theta,x\rangle)$ | any | **No** (not even strictly convex) |
| $\log z$ (in portfolio) | $z \in $ bounded simplex | Yes |
| $\|x\|_1$ | any | **No** |

## Geometric intuition

"The function curves up faster than its gradient grows." Concretely: if you're at a point where the gradient is large (meaning you're far from a local minimum), then the second-order curvature is also large, so a second-order step makes meaningful progress. Convex-but-not-exp-concave functions can be "flat with steep slopes" in a way that defeats Newton-like methods.

The squared-loss case is the prototype: $\nabla^2 f = xx^\top$ and $\nabla f = -2(y - \langle\theta, x\rangle) x$, so the condition $xx^\top \succeq \alpha \cdot 4(y - \hat y)^2 \cdot xx^\top$ holds when $4\alpha(y - \hat y)^2 \leq 1$, i.e. on bounded data.

## Where it shows up in this course

- **[[Online Ridge Regression]]** — exp-concavity gives the $O(\log T)$ regret.
- **Online Newton Step (ONS)** — the algorithm exploiting exp-concavity.
- Implicitly behind **[[LinUCB]]** (same Gram-matrix machinery).
- the intro lecture (slide 18) flags it as the reason online ridge has stronger guarantees than OGD.

## References

- Hazan 2022, §4.2 — full treatment of ONS.
- Hazan, Agarwal, Kale 2007 — original analysis.
- Cesa-Bianchi & Lugosi, *Prediction, Learning, and Games*, Ch. 11.

## See also

- [[Convexity]] — parent concept.
- [[Online Ridge Regression]] — main application.
- [[Online Convex Optimization]] — framework.
