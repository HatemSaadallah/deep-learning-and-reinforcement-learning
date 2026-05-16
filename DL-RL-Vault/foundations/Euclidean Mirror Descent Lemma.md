# Euclidean Mirror Descent Lemma

**Tags:** #foundational #lemma #online-learning

The one-step regret inequality for [[OGD Regret Bound|online gradient descent]] (= Euclidean mirror descent). Together with telescoping, it gives the $O(\sqrt{T})$ OGD bound the way the lecture proves it.

## Statement

Let $\mathcal{C} \subseteq \mathbb{R}^d$ be convex, $\theta^* \in \mathcal{C}$, and let $\theta_{t+1} = \Pi_\mathcal{C}(\theta_t - \eta g_t)$ for some $g_t \in \mathbb{R}^d$. Then
$$\boxed{\;\langle g_t,\, \theta_t - \theta^*\rangle \;\leq\; \frac{\|\theta_t - \theta^*\|^2 - \|\theta_{t+1} - \theta^*\|^2}{2\eta} \;+\; \frac{\eta}{2}\|g_t\|^2.\;}$$

In words: the per-round "regret contribution" $\langle g_t, \theta_t - \theta^*\rangle$ is bounded by **(a)** a telescoping potential drop, plus **(b)** a step-size noise term.

## Proof

**Step 1 — Projection is contractive toward $\mathcal{C}$.** Since $\theta^* \in \mathcal{C}$ and $\Pi_\mathcal{C}$ projects onto a convex set:
$$\|\theta_{t+1} - \theta^*\|^2 \;=\; \|\Pi_\mathcal{C}(\theta_t - \eta g_t) - \theta^*\|^2 \;\leq\; \|\theta_t - \eta g_t - \theta^*\|^2.$$

**Step 2 — Expand the squared norm:**
$$\|\theta_t - \eta g_t - \theta^*\|^2 \;=\; \|\theta_t - \theta^*\|^2 - 2\eta\langle g_t, \theta_t - \theta^*\rangle + \eta^2 \|g_t\|^2.$$

**Step 3 — Combine and rearrange:**
$$2\eta\langle g_t, \theta_t - \theta^*\rangle \;\leq\; \|\theta_t - \theta^*\|^2 - \|\theta_{t+1} - \theta^*\|^2 + \eta^2 \|g_t\|^2.$$

Divide by $2\eta$. $\square$

## Connection to the [[Three-point Equality]]

The Euclidean version of $D_R = \tfrac{1}{2}\|\cdot - \cdot\|^2$ in the three-point identity gives:
$$\frac{1}{2}\|\theta^* - \theta_t\|^2 - \frac{1}{2}\|\theta^* - \theta_{t+1}\|^2 - \frac{1}{2}\|\theta_{t+1} - \theta_t\|^2 = \langle \theta_{t+1} - \theta_t, \theta_{t+1} - \theta^*\rangle.$$

Combined with the OGD optimality condition $\langle \theta_{t+1} - \theta_t + \eta g_t, u - \theta_{t+1}\rangle \geq 0$ for $u = \theta^*$, this re-derives the lemma. So the **Euclidean MD lemma is a corollary of the three-point identity** + projection optimality.

## How it gives the $RL\sqrt{T}$ OGD regret (lecture-style proof)

By convexity, $\ell_t(\theta_t) - \ell_t(\theta^*) \leq \langle g_t, \theta_t - \theta^*\rangle$. Sum the lemma over $t = 1, \dots, T$:
$$R_T \;\leq\; \sum_t \langle g_t, \theta_t - \theta^*\rangle \;\leq\; \frac{1}{2\eta}\sum_t \bigl[\|\theta_t - \theta^*\|^2 - \|\theta_{t+1} - \theta^*\|^2\bigr] + \frac{\eta}{2}\sum_t \|g_t\|^2.$$

The first sum **telescopes** to $\tfrac{1}{2\eta}\bigl[\|\theta_1 - \theta^*\|^2 - \|\theta_{T+1} - \theta^*\|^2\bigr] \leq R^2/(2\eta)$ (with $\mathcal{C} \subseteq B(\theta_1, R)$).
The second sum is $\leq \eta L^2 T / 2$ (with $\|g_t\| \leq L$).
$$R_T \leq \frac{R^2}{2\eta} + \frac{\eta L^2 T}{2}.$$
Optimal $\eta = R/(L\sqrt{T})$ gives $R_T \leq RL\sqrt{T}$. $\square$

This is the form the lecture proves on slides 26-28 of [[Lecture 01 - Intro|the intro lecture]]. The exact same rate is obtained by the alternative proof in [[OGD Regret Bound]] via direct telescoping.

## See also

- [[OGD Regret Bound]] — the regret bound this lemma serves.
- [[Three-point Equality]] — the general identity this specializes.
- [[Mirror Descent Analysis]] — the general framework.
- [[Bregman Divergence]] — the geometry being specialized to Euclidean.
