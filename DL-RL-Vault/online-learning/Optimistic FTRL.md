# Optimistic FTRL

**Tags:** #online-learning #algorithm #game-theory

A variant of [[Follow the Regularized Leader|FTRL]] that uses a **prediction $m_t$** of the next loss to act one step ahead. When predictions are accurate, regret can be much lower than the worst-case $O(\sqrt{T})$ — down to $O(\log T)$ or even $O(1)$ in favorable cases.

The key application is **self-play in games**: when both players use Optimistic FTRL, the saddle-point gap shrinks at $O(1/T)$ rather than the $O(1/\sqrt{T})$ of vanilla FTRL.

## Setting

Standard [[Online Convex Optimization|OCO]]: at round $t$ choose $x_t \in \mathcal{C}$, observe loss $\ell_t$, suffer $\ell_t(x_t)$. Additionally:
- **Prediction $m_t$** available at the start of round $t$ — an approximation of what $\ell_t$ will be.

## Algorithm

$$x_t \;=\; \arg\min_{x \in \mathcal{C}} \left\langle x,\; L_{t-1} + m_t\right\rangle + \frac{1}{\eta}\psi(x)$$
where $L_{t-1} = \sum_{\tau < t} \ell_\tau$ is the cumulative loss so far. Compared to vanilla FTRL ($x_t = \arg\min \langle x, L_{t-1}\rangle + \psi(x)/\eta$), the extra term $m_t$ **shifts the leader** as if we'd already seen the prediction.

**Typical choice of prediction:** $m_t = \ell_{t-1}$ — "the last loss is a good guess for the next."

## RVU (Regret Bounded by Variation in Utilities) property

Syrgkanis et al. 2015. For player $i$ using Optimistic FTRL with prediction $m_t = \ell_{t-1}^{(i)}$:
$$\boxed{\;R_T^{(i)} \;\leq\; \underbrace{\frac{2 + \log K}{\eta}}_{\text{regularization}} \;+\; \underbrace{\eta \sum_t \|\ell_t - \ell_{t-1}\|_\infty^2}_{\text{path-length of losses}} \;-\; \underbrace{\frac{1}{4\eta}\sum_t \|p_{t+1} - p_t\|_1^2}_{\text{algorithm stability}}.\;}$$

**Interpretation:**
- Quadratic in **how much the loss sequence varies between consecutive rounds**.
- **Subtracts** the squared movement of your own strategy — so stable players get credit.
- Recovers Hedge's $O(\sqrt{T \log K})$ in the worst case (when $\|\ell_t - \ell_{t-1}\|$ is constant), but can be **much smaller** when losses are slowly-varying or predictable.

## $O(1/T)$ convergence in self-play (zero-sum games)

Setup: two players run Optimistic FTRL with $m_t^{(i)} = \ell_{t-1}^{(i)}$ on payoff matrix $A$. Then $\ell_t$ for player 1 depends on $y_t$ (player 2's strategy), and vice versa. The "loss variation" $\|\ell_t - \ell_{t-1}\|$ is bounded by the **opponent's strategy movement** $\|y_t - y_{t-1}\|_1$.

Plugging the RVU bounds for both players (with same $\eta$):
$$R_T^{(1)} + R_T^{(2)} \;\leq\; \frac{2(2 + \log K)}{\eta} + 2\eta + \left(\eta - \frac{1}{4\eta}\right) \sum_t \bigl[\|x_t - x_{t-1}\|_1^2 + \|y_t - y_{t-1}\|_1^2\bigr].$$

Setting $\eta = 1/2$ kills the last sum (coefficient $\eta - 1/(4\eta) = 0$):
$$R_T^{(1)} + R_T^{(2)} \;\leq\; O(1).$$

**Constant** sum of regrets. By the [[Regret and Equilibria|regret-to-equilibrium reduction]] $\gamma(\bar x_T, \bar y_T) \leq (R_T^{(1)} + R_T^{(2)})/T$:
$$\gamma(\bar x_T, \bar y_T) \;=\; O(1/T).$$

A quadratic improvement over vanilla FTRL's $O(1/\sqrt{T})$.

## Why optimism beats pessimism here

Vanilla FTRL is **pessimistic** — it assumes nothing about the next loss, leading to a $\sqrt{T}$ "concentration tax." In self-play, the opponent's strategy changes slowly when both players are stable, so the losses *are* predictable. Optimistic FTRL exploits this: by anticipating $\ell_t \approx \ell_{t-1}$, it avoids the per-round noise that vanilla FTRL pays for.

The negative term $-\sum \|p_{t+1} - p_t\|^2$ in the RVU bound is the formal expression of this: the more stable your *own* play, the smaller your regret.

## Two notions of convergence

| Metric | Vanilla FTRL | Optimistic FTRL |
|---|---|---|
| Average iterate $\gamma(\bar x_T, \bar y_T)$ | $O(1/\sqrt{T})$ | $O(1/T)$ |
| Last iterate $\gamma(x_T, y_T)$ | does **not** converge (cycles) | converges (empirically; theory case-by-case) |

The last-iterate convergence is striking visually: vanilla FTRL spins around the equilibrium forever; Optimistic FTRL spirals in.

## See also

- [[Follow the Regularized Leader]] — the vanilla version.
- [[Regret and Equilibria]] — the regret → Nash reduction this enables.
- [[Nash Equilibrium]] — the target.
- [[Mirror Descent Analysis]] — Optimistic Mirror Descent is the equivalent in the MD framework.

## References

- Syrgkanis, Agarwal, Luo, Schapire, *Fast convergence of regularized learning in games*, NeurIPS 2015.
- Rakhlin & Sridharan, *Online learning with predictable sequences*, COLT 2013 (origin of the optimistic-FTRL framework).
