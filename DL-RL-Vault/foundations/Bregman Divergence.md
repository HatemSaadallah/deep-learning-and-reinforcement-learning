# Bregman Divergence

**Tags:** #foundational #convex-analysis #definition

A generalization of squared Euclidean distance, parameterized by a strictly convex "regularizer" $R$. Acts as the **local geometry** for [[Mirror Descent Analysis|mirror descent]] and [[Follow the Regularized Leader|FTRL]].

## Definition

Let $R: \mathcal{C} \to \mathbb{R}$ be a continuously differentiable, strictly convex function on a convex set $\mathcal{C}$. The **Bregman divergence** induced by $R$ is
$$D_R(x, y) \;:=\; R(x) - R(y) - \langle \nabla R(y),\, x - y\rangle.$$

**Geometric reading:** $D_R(x, y)$ is the gap between $R(x)$ and its first-order Taylor approximation around $y$. It measures how much $R$ "curves up" from $y$ in the direction of $x$.

## Properties

1. **Non-negativity:** $D_R(x, y) \geq 0$, with equality iff $x = y$ (when $R$ is strictly convex).

2. **Asymmetric:** in general $D_R(x, y) \neq D_R(y, x)$. So it is **not a metric** — no triangle inequality.

3. **Convex in the first argument** (for fixed $y$). Not generally convex in the second.

4. **Strong convexity ⇔ lower bound:** $R$ is $\sigma$-strongly convex w.r.t. $\|\cdot\|$ iff
$$D_R(x, y) \;\geq\; \frac{\sigma}{2}\|x - y\|^2 \qquad \forall x, y \in \mathcal{C}.$$
This is the **load-bearing fact** in regret bounds via FTRL / MD.

5. **Three-point identity (the workhorse lemma):**
$$\boxed{\;D_R(u, v) - D_R(u, w) - D_R(w, v) \;=\; \langle \nabla R(w) - \nabla R(v),\; w - u\rangle.\;}$$
*Verify by expanding both sides.* This generalizes the law of cosines.

6. **Mirror-descent first-order optimality:** if $x_{t+1} = \arg\min_{x \in \mathcal{C}}\bigl[\eta\langle g, x\rangle + D_R(x, x_t)\bigr]$, then
$$\langle \nabla R(x_{t+1}) - \nabla R(x_t) + \eta g,\, u - x_{t+1}\rangle \geq 0 \quad \forall u \in \mathcal{C}.$$
Combined with the three-point identity, this gives the standard MD regret analysis.

## Canonical examples

| Regularizer $R(x)$ | Domain | Bregman $D_R(x, y)$ | Name |
|---|---|---|---|
| $\tfrac{1}{2}\|x\|_2^2$ | $\mathbb{R}^d$ | $\tfrac{1}{2}\|x - y\|_2^2$ | Euclidean squared distance |
| $\sum_i x_i \log x_i - x_i$ | $\mathbb{R}^d_{>0}$ | $\sum_i \bigl[x_i \log(x_i/y_i) - x_i + y_i\bigr]$ | **Generalized KL** (unnormalized) |
| $\sum_i x_i \log x_i$ on simplex $\Delta_N$ | $\Delta_N$ | $\sum_i x_i \log(x_i/y_i)$ | **[[KL Divergence]]** |
| $-\sum_i \log x_i$ | $\mathbb{R}^d_{>0}$ | $\sum_i \bigl[x_i/y_i - \log(x_i/y_i) - 1\bigr]$ | Itakura-Saito |
| $\tfrac{1}{2} x^\top A x$ for $A \succ 0$ | $\mathbb{R}^d$ | $\tfrac{1}{2}(x-y)^\top A (x-y)$ | Mahalanobis |

Special-case check: $R(x) = \tfrac12\|x\|^2 \Rightarrow \nabla R(y) = y \Rightarrow D_R(x, y) = \tfrac12\|x\|^2 - \tfrac12\|y\|^2 - \langle y, x-y\rangle = \tfrac12\|x-y\|^2$. ✓

The KL case is why **mirror descent with negative entropy on the simplex recovers [[Hedge - Multiplicative Weights|Hedge]]** — see [[Mirror Descent Analysis]].

## Where Bregman divergences appear in the course

- **[[Mirror Descent Analysis|Mirror descent]]:** the update is $x_{t+1} = \arg\min_x \eta\langle g_t, x\rangle + D_R(x, x_t)$.
- **[[Follow the Regularized Leader|FTRL]] (proximal form):** equivalent to MD via the three-point identity.
- **Three-point equality** in OGD analysis (Euclidean special case).
- **[[KL Divergence]]:** a Bregman divergence (with $R$ = negative entropy). Many information-theoretic inequalities specialize to this case.

## Why it's the right notion of "distance"

Euclidean distance $\|x - y\|^2$ implicitly assumes uniform geometry. On the simplex, on $\mathbb{R}^d_{>0}$, on PSD matrices, etc., this is unnatural — distributions near the boundary should feel "far" from those near the center. Bregman divergences let the regularizer $R$ encode the geometry: small KL divergence means the distributions are operationally hard to distinguish, regardless of their $\ell_2$ distance.

This is why mirror descent with the right $R$ beats OGD on geometries like the simplex: $\log N$ instead of $N$ in regret bounds.

## See also

- [[KL Divergence]] — the most important Bregman divergence in ML.
- [[Pinskers Inequality]] — converts KL (a Bregman) to total variation.
- [[Mirror Descent Analysis]] — the main place Bregman shows up algorithmically.
- [[Follow the Regularized Leader]] — equivalent formulation.
