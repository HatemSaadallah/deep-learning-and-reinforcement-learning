# Proof — OGD Regret $O(\sqrt{T})$

**Tags:** #proof #online-learning #ogd
**Topic:** Regret guarantees of online gradient descent.

## Statement

Let $\mathcal{K}$ be a convex compact set with diameter $D$, and let $f_1, \dots, f_T$ be convex functions with subgradients bounded by $\|g_t\|_2 \leq G$. **Online gradient descent (OGD)** plays
$$x_{t+1} = \Pi_\mathcal{K}(x_t - \eta\, g_t), \qquad g_t \in \partial f_t(x_t).$$
With $\eta = \dfrac{D}{G\sqrt{T}}$,
$$\boxed{\;R_T = \sum_{t=1}^T f_t(x_t) - \min_{x \in \mathcal{K}} \sum_{t=1}^T f_t(x) \;\leq\; DG\sqrt{T}.\;}$$

## Setup

- $\mathcal{K} \subset \mathbb{R}^d$ convex, $\|x - y\|_2 \leq D$ for all $x, y \in \mathcal{K}$.
- $\Pi_\mathcal{K}$: Euclidean projection onto $\mathcal{K}$.
- $x^* \in \arg\min_{x \in \mathcal{K}} \sum_t f_t(x)$.
- Notation: $D_t := \|x_t - x^*\|_2^2$.

## Proof

**Step 1 — One-step inequality (projection contracts).**

Since $x^* \in \mathcal{K}$ and projection onto a convex set contracts distances to points inside it:
$$D_{t+1} = \|\Pi_\mathcal{K}(x_t - \eta g_t) - x^*\|_2^2 \;\leq\; \|x_t - \eta g_t - x^*\|_2^2.$$

Expanding the RHS:
$$D_{t+1} \leq D_t - 2\eta\langle g_t, x_t - x^*\rangle + \eta^2 \|g_t\|_2^2.$$

Rearranging:
$$\langle g_t, x_t - x^*\rangle \;\leq\; \frac{D_t - D_{t+1}}{2\eta} + \frac{\eta}{2}\|g_t\|_2^2.$$

**Step 2 — Convexity.**

By definition of subgradient (or convexity of $f_t$):
$$f_t(x_t) - f_t(x^*) \;\leq\; \langle g_t, x_t - x^*\rangle.$$

Combine with Step 1:
$$f_t(x_t) - f_t(x^*) \;\leq\; \frac{D_t - D_{t+1}}{2\eta} + \frac{\eta G^2}{2}.$$

**Step 3 — Sum and telescope.**

$$R_T = \sum_{t=1}^T \bigl[f_t(x_t) - f_t(x^*)\bigr] \;\leq\; \frac{D_1 - D_{T+1}}{2\eta} + \frac{\eta G^2 T}{2} \;\leq\; \frac{D^2}{2\eta} + \frac{\eta G^2 T}{2}.$$

(Using $D_1 \leq D^2$ since $x_1, x^* \in \mathcal{K}$, and $D_{T+1} \geq 0$.)

**Step 4 — Optimize $\eta$.**

Minimize the RHS over $\eta > 0$: setting the derivative to zero,
$$-\frac{D^2}{2\eta^2} + \frac{G^2 T}{2} = 0 \;\Longrightarrow\; \eta^* = \frac{D}{G\sqrt{T}}.$$

Plugging in:
$$R_T \leq \frac{D^2}{2} \cdot \frac{G\sqrt{T}}{D} + \frac{D}{G\sqrt{T}} \cdot \frac{G^2 T}{2} = \frac{DG\sqrt{T}}{2} + \frac{DG\sqrt{T}}{2} = DG\sqrt{T}. \qquad\square$$

## Intuition / what to remember

- **Telescope a potential function.** Here the potential is $\|x_t - x^*\|_2^2$. Its one-step change relates the gradient inner product to a clean telescoping sum.
- **Two error terms:** $D^2/(2\eta)$ ("distance to start") and $\eta G^2 T / 2$ ("step-size noise"). The optimal $\eta$ balances them — both end up $\sim DG\sqrt{T}/2$.
- **The proof never uses smoothness.** Just convexity and bounded gradients. So OGD works for non-smooth losses (e.g. hinge, $\ell_1$).
- **Anytime version:** if $T$ unknown, take $\eta_t = D/(G\sqrt{t})$, regret still $O(DG\sqrt{T})$ (up to a constant).
- **OGD is a special case of [[Follow the Regularized Leader|FTRL]]** with $R(x) = \tfrac{1}{2}\|x\|^2$ and of [[Mirror Descent Analysis|mirror descent]] with the same regularizer.

## Lecture-style variant of the proof (lecture slides 26–28)

the intro lecture proves the same theorem via two **named lemmas**:

1. **[[Three-point Equality]]** specialized to $R(x) = \tfrac12\|x\|_2^2$ (i.e. the parallelogram-law identity).
2. **[[Euclidean Mirror Descent Lemma]]** — the one-step inequality $\langle g_t, \theta_t - \theta^*\rangle \leq \tfrac{1}{2\eta}(\|\theta_t - \theta^*\|^2 - \|\theta_{t+1} - \theta^*\|^2) + \tfrac{\eta}{2}\|g_t\|^2$.

The direct telescoping above (Steps 1–3) **is** the Euclidean MD lemma in disguise. If the exam asks "prove the regret of OGD using the Euclidean MD lemma", cite the lemma, sum its conclusion over $t$, telescope the first sum to get $R^2/(2\eta)$, and bound the second by $\eta L^2 T / 2$. Same end result, presented in the form the lecture expects.

The lecture states the bound with $R$ = radius of a ball $B(\theta_1, R)$ containing $\mathcal{C}$ (rather than the diameter $D$), and uses $L$ for the gradient bound (instead of $G$). Optimizing $\eta = R/(L\sqrt{T})$ gives $R_T \leq RL\sqrt{T}$ — same rate, slightly different constants.

## See also

- [[Follow the Regularized Leader]] — the general framework OGD lives in.
- [[Mirror Descent Analysis]] — same proof template with Bregman divergences.
- [[Three-point Equality]] / [[Euclidean Mirror Descent Lemma]] — the named lemmas the lecture uses.
- [[Regret]] — the quantity being bounded.
