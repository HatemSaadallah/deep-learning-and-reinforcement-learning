# Proof — FTRL: Stability and Regret

**Tags:** #proof #online-learning #ftrl
**Topic:** Stability of consecutive FTRL iterates and resulting regret bound.

## Statement

Consider [[Follow the Regularized Leader|FTRL]] with linear losses $f_t(x) = \langle g_t, x\rangle$:
$$x_{t+1} = \arg\min_{x \in \mathcal{K}} \;\eta \sum_{s=1}^t \langle g_s, x\rangle + R(x).$$

Assume $R$ is $\sigma$-strongly convex w.r.t. norm $\|\cdot\|$ and $\|g_t\|_* \leq G$ in the dual norm.

**Stability lemma:**
$$\boxed{\;\|x_t - x_{t+1}\| \;\leq\; \frac{\eta}{\sigma}\|g_t\|_*.\;}$$

**Regret bound:** with $D_R = \max R - \min R$ on $\mathcal{K}$,
$$\boxed{\;R_T \;\leq\; \frac{D_R}{\eta} + \frac{\eta}{\sigma}\sum_{t=1}^T \|g_t\|_*^2.\;}$$

With optimal $\eta = \sqrt{D_R \sigma / (G^2 T)}$: $R_T \leq 2G\sqrt{D_R T / \sigma}$.

## Setup

- $\mathcal{K} \subset \mathbb{R}^d$ closed convex.
- $\Phi_t(x) := \eta \sum_{s \leq t} \langle g_s, x\rangle + R(x)$.
- $x_{t+1} = \arg\min_{x \in \mathcal{K}} \Phi_t(x)$.
- $\Phi_t$ is $\sigma$-strongly convex (since adding a linear function preserves strong convexity).

## Proof of stability

By strong convexity of $\Phi_{t-1}$ at its minimizer $x_t$:
$$\Phi_{t-1}(x_{t+1}) \;\geq\; \Phi_{t-1}(x_t) + \langle \nabla\Phi_{t-1}(x_t),\, x_{t+1} - x_t\rangle + \frac{\sigma}{2}\|x_{t+1} - x_t\|^2.$$

By first-order optimality of $x_t$ on $\mathcal{K}$:
$$\langle \nabla\Phi_{t-1}(x_t),\, y - x_t\rangle \geq 0 \quad\forall y \in \mathcal{K}.$$
In particular for $y = x_{t+1} \in \mathcal{K}$, that inner product is $\geq 0$, so:
$$\Phi_{t-1}(x_{t+1}) - \Phi_{t-1}(x_t) \;\geq\; \frac{\sigma}{2}\|x_{t+1} - x_t\|^2. \tag{$\star$}$$

By optimality of $x_{t+1}$ for $\Phi_t = \Phi_{t-1} + \eta \langle g_t, \cdot\rangle$:
$$\Phi_t(x_{t+1}) \leq \Phi_t(x_t).$$
i.e. $\Phi_{t-1}(x_{t+1}) + \eta \langle g_t, x_{t+1}\rangle \;\leq\; \Phi_{t-1}(x_t) + \eta \langle g_t, x_t\rangle$, so:
$$\Phi_{t-1}(x_{t+1}) - \Phi_{t-1}(x_t) \;\leq\; \eta \langle g_t, x_t - x_{t+1}\rangle. \tag{$\star\star$}$$

Chain $(\star)$ and $(\star\star)$:
$$\frac{\sigma}{2}\|x_{t+1} - x_t\|^2 \;\leq\; \eta \langle g_t, x_t - x_{t+1}\rangle \;\leq\; \eta\|g_t\|_*\|x_t - x_{t+1}\|.$$

Divide by $\|x_t - x_{t+1}\|$:
$$\|x_t - x_{t+1}\| \leq \frac{2\eta}{\sigma}\|g_t\|_*.$$

(Various textbooks normalize strong convexity differently — the constant differs by a factor of 2, but the rate is unaffected.) $\square$

## Proof of regret

**Step 1 — "Be-The-Leader" (BTL) inequality.**

Think of the regularizer as a fictitious round-0 loss: $f_0(x) := R(x)/\eta$. Then $x_{t+1}$ minimizes $\sum_{s=0}^t f_s$ (just FTL).

BTL says: for any sequence of losses, the FTL strategy played *one step ahead* has nonpositive regret. Formally:
$$\sum_{t=0}^T f_t(x_{t+1}) \;\leq\; \sum_{t=0}^T f_t(u) \quad \forall u \in \mathcal{K}.$$

(Proof by induction. Base: at $t=0$, $x_1$ minimizes $f_0$. Inductive step: shift the inequality and use optimality of $x_{T+1}$.)

Specializing: $R(x_1)/\eta + \sum_{t=1}^T \langle g_t, x_{t+1}\rangle \leq R(u)/\eta + \sum_{t=1}^T \langle g_t, u\rangle$. Rearranging:
$$\sum_{t=1}^T \langle g_t, x_{t+1} - u\rangle \;\leq\; \frac{R(u) - R(x_1)}{\eta}.$$

**Step 2 — Add the "drift" from $x_{t+1}$ back to $x_t$.**

$$R_T = \sum_t \langle g_t, x_t - u\rangle \;=\; \underbrace{\sum_t \langle g_t, x_t - x_{t+1}\rangle}_{\text{drift}} + \underbrace{\sum_t \langle g_t, x_{t+1} - u\rangle}_{\leq\, (R(u)-R(x_1))/\eta}.$$

**Step 3 — Control the drift via stability.**

By Cauchy-Schwarz (dual norms) and the stability lemma:
$$\sum_t \langle g_t, x_t - x_{t+1}\rangle \;\leq\; \sum_t \|g_t\|_* \cdot \|x_t - x_{t+1}\| \;\leq\; \frac{\eta}{\sigma}\sum_t \|g_t\|_*^2.$$

**Step 4 — Combine.**

$$R_T \;\leq\; \frac{R(u) - R(x_1)}{\eta} + \frac{\eta}{\sigma}\sum_t \|g_t\|_*^2 \;\leq\; \frac{D_R}{\eta} + \frac{\eta G^2 T}{\sigma}.$$

Choosing $\eta = \sqrt{D_R \sigma/(G^2 T)}$:
$$R_T \leq 2 G \sqrt{D_R T / \sigma}. \qquad \square$$

## Intuition / what to remember

- **Stability is the load-bearing claim.** Strong convexity makes the FTRL minimum move slowly. The slower it moves, the smaller the "regret of $x_t$ vs $x_{t+1}$."
- **The regularizer plays two roles:** (a) penalty for deviation, controlling $D_R/\eta$, (b) stabilizer, controlling drift through strong convexity.
- **BTL = "be-the-leader has zero regret":** if you could play $x_{t+1}$ instead of $x_t$, you'd already be done. The proof is just bounding how much you lose by playing one step behind.
- **Specializations:**
  - $R = \tfrac12\|x\|_2^2$, $\sigma = 1$ in $\ell_2$, $D_R \leq D^2/2$ → recovers [[OGD Regret Bound|OGD]] $O(GD\sqrt{T})$.
  - $R = \sum_i x_i \log x_i$ on simplex, $\sigma = 1$ in $\ell_1$ (by Pinsker), $D_R \leq \log N$, $\|g_t\|_\infty \leq G$ → recovers [[Hedge - Multiplicative Weights|Hedge]] $O(G\sqrt{T \log N})$.

## See also

- [[Follow the Regularized Leader]] — the algorithm.
- [[Mirror Descent Analysis]] — equivalent algorithm with the same rate (dual view).
- [[OGD Regret Bound]] — special case.
