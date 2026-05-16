# Proof — Mirror Descent and OMD on the Simplex

**Tags:** #proof #online-learning #mirror-descent
**Topic:** Regret analysis of (online) mirror descent and its specialization to the simplex.

## Statement

**Online Mirror Descent (OMD)** on convex $\mathcal{K}$ with $\sigma$-strongly convex regularizer $R$:
$$x_{t+1} = \arg\min_{x \in \mathcal{K}} \;\eta \langle g_t, x\rangle + D_R(x, x_t)$$
where $D_R(x, y) := R(x) - R(y) - \langle \nabla R(y), x - y\rangle$ is the **Bregman divergence**.

**Regret bound:**
$$\boxed{\;R_T \;\leq\; \frac{D_R(u, x_1)}{\eta} + \frac{\eta}{2\sigma}\sum_{t=1}^T \|g_t\|_*^2.\;}$$

Optimal $\eta$ gives $R_T = O(G\sqrt{D_R T/\sigma})$ — same rate as [[FTRL Stability and Regret|FTRL]].

**Simplex case** (negative-entropy $R$): recovers [[Hedge - Multiplicative Weights|Hedge]] with regret $O(G\sqrt{T \log N})$.

## Setup

Two key facts about Bregman divergences:
1. $D_R \geq 0$, $= 0 \iff x = y$.
2. **Three-point identity (the load-bearing tool):**
$$D_R(u, v) - D_R(u, w) - D_R(w, v) \;=\; \langle \nabla R(w) - \nabla R(v),\, w - u\rangle. \tag{3PT}$$

Strong convexity of $R$ implies: $D_R(x, y) \geq \tfrac{\sigma}{2}\|x - y\|^2$.

## Proof

**Step 1 — First-order optimality of $x_{t+1}$.**

The variational inequality for the OMD update on $\mathcal{K}$:
$$\bigl\langle\, \eta g_t + \nabla R(x_{t+1}) - \nabla R(x_t),\; u - x_{t+1}\bigr\rangle \;\geq\; 0 \quad \forall u \in \mathcal{K}.$$
Rearrange:
$$\langle \nabla R(x_{t+1}) - \nabla R(x_t),\, x_{t+1} - u\rangle \;\leq\; \eta \langle g_t, u - x_{t+1}\rangle. \tag{$\dagger$}$$

**Step 2 — Apply three-point identity.**

Set $v = x_t, w = x_{t+1}$ in (3PT):
$$\langle \nabla R(x_{t+1}) - \nabla R(x_t),\, x_{t+1} - u\rangle \;=\; D_R(u, x_t) - D_R(u, x_{t+1}) - D_R(x_{t+1}, x_t).$$

Combine with $(\dagger)$:
$$D_R(u, x_t) - D_R(u, x_{t+1}) - D_R(x_{t+1}, x_t) \;\leq\; \eta \langle g_t, u - x_{t+1}\rangle.$$

Equivalently:
$$\eta \langle g_t, x_{t+1} - u\rangle \;\leq\; D_R(u, x_t) - D_R(u, x_{t+1}) - D_R(x_{t+1}, x_t). \tag{$\ddagger$}$$

**Step 3 — Decompose into played-action regret + drift.**

$$\eta \langle g_t, x_t - u\rangle \;=\; \eta \langle g_t, x_{t+1} - u\rangle + \eta \langle g_t, x_t - x_{t+1}\rangle.$$

Plug in $(\ddagger)$ for the first term:
$$\eta \langle g_t, x_t - u\rangle \leq \underbrace{D_R(u, x_t) - D_R(u, x_{t+1})}_{\text{telescoping}} \underbrace{- D_R(x_{t+1}, x_t) + \eta\langle g_t, x_t - x_{t+1}\rangle}_{\text{controlled by stability}}.$$

**Step 4 — Control the stability term.**

Use strong convexity $D_R(x_{t+1}, x_t) \geq \tfrac{\sigma}{2}\|x_{t+1} - x_t\|^2$ and the Fenchel/Young inequality:
$$\eta \langle g_t, x_t - x_{t+1}\rangle - \tfrac{\sigma}{2}\|x_t - x_{t+1}\|^2 \;\leq\; \frac{\eta^2}{2\sigma}\|g_t\|_*^2.$$

(Maximize $\eta\|g_t\|_* z - \tfrac{\sigma}{2}z^2$ over $z = \|x_t - x_{t+1}\| \geq 0$: optimum at $z = \eta\|g_t\|_*/\sigma$ giving max $\eta^2\|g_t\|_*^2/(2\sigma)$.)

**Step 5 — Sum and telescope.**

$$\eta\sum_t \langle g_t, x_t - u\rangle \;\leq\; \underbrace{D_R(u, x_1) - D_R(u, x_{T+1})}_{\leq\, D_R(u, x_1)} + \frac{\eta^2}{2\sigma}\sum_t \|g_t\|_*^2.$$

Divide by $\eta$:
$$R_T \leq \sum_t \langle g_t, x_t - u\rangle \;\leq\; \frac{D_R(u, x_1)}{\eta} + \frac{\eta}{2\sigma}\sum_t \|g_t\|_*^2. \qquad \square$$

## OMD on the simplex = Hedge

**Setup:** $\mathcal{K} = \Delta_N$ (probability simplex), $R(x) = \sum_i x_i \log x_i$ (negative entropy).

**Bregman:** $D_R(x, y) = \sum_i x_i \log(x_i/y_i) = \mathrm{KL}(x, y)$.

**Strong convexity:** by [[Pinskers Inequality]], $\mathrm{KL}(x, y) \geq \tfrac{1}{2}\|x - y\|_1^2$, so $\sigma = 1$ w.r.t. $\|\cdot\|_1$. The dual norm is $\|\cdot\|_\infty$.

**Update rule:** the OMD step on the simplex has closed form. Lagrangian (with multipliers for $\sum x_i = 1$):
$$\eta g_t(i) + \log\frac{x(i)}{x_t(i)} + \lambda = 0$$
$$x_{t+1}(i) \;\propto\; x_t(i)\, e^{-\eta g_t(i)}.$$
Normalize to sum 1. **This is exactly Hedge / exponential weights / multiplicative weights.**

**Regret bound:** with $x_1 =$ uniform, $D_R(u, x_1) = \log N - H(u) \leq \log N$. With $\|g_t\|_\infty \leq G$:
$$R_T \leq \frac{\log N}{\eta} + \frac{\eta T G^2}{2}.$$
Optimal $\eta = \sqrt{2 \log N/(T G^2)}$:
$$R_T \leq G\sqrt{2T \log N}.$$

This matches Hedge's known regret — logarithmic in $N$ thanks to entropy regularization.

## Intuition / what to remember

- **Mirror descent generalizes gradient descent to non-Euclidean geometries.** The Bregman divergence is the local metric.
- **The three-point identity replaces the "expand $\|x_t - x^*\|^2$" telescoping** from the [[OGD Regret Bound|OGD proof]]. Same role, more general.
- **MD ≡ FTRL** for the linear-loss case (just different parameterizations of the same iterates). Same regret rate.
- **Entropy regularizer on simplex gives multiplicative weights.** The "mirror" is the log map; the dual update is additive on the log probabilities, which becomes multiplicative on the probabilities.
- **The dual norm matters.** For simplex problems use $\ell_1/\ell_\infty$, not $\ell_2/\ell_2$ — that's how $\log N$ instead of $N$ appears.

## See also

- [[Follow the Regularized Leader]] — the equivalent algorithm.
- [[FTRL Stability and Regret]] — alternative analysis route.
- [[Hedge - Multiplicative Weights]] — the simplex instance.
- [[Pinskers Inequality]] — gives the strong-convexity constant for entropy.
