# Proof — TRPO Surrogate Objective

**Tags:** #proof #policy-gradient #trpo
**Topic:** Derivation of the TRPO surrogate objective from the performance difference lemma.

## Statement

Let $\pi$ be the current policy and $\pi'$ a candidate next policy. Define the **surrogate**:
$$L_\pi(\pi') \;:=\; J(\pi) + \frac{1}{1-\gamma}\,\mathbb{E}_{s \sim d^\pi_\gamma,\, a \sim \pi(\cdot|s)}\!\left[\frac{\pi'(a|s)}{\pi(a|s)}\,A^\pi(s, a)\right].$$

**Performance Difference Lemma (PDL, Kakade & Langford 2002):**
$$J(\pi') - J(\pi) \;=\; \frac{1}{1-\gamma}\,\mathbb{E}_{s \sim d^{\pi'}_\gamma,\, a \sim \pi'(\cdot|s)}\!\bigl[A^\pi(s, a)\bigr].$$

**Approximation bound (Schulman et al. 2015):**
$$\bigl|J(\pi') - L_\pi(\pi')\bigr| \;\leq\; \frac{4 \varepsilon\, \gamma}{(1-\gamma)^2}\, D_{\text{KL}}^{\max}(\pi, \pi'),$$
where $\varepsilon = \max_{s,a}|A^\pi(s,a)|$ and $D_{\text{KL}}^{\max}(\pi, \pi') = \max_s \mathrm{KL}\bigl(\pi(\cdot|s) \,\|\, \pi'(\cdot|s)\bigr)$.

**TRPO:** maximize $L_\pi(\pi')$ subject to $D_{\text{KL}}^{\max}(\pi, \pi') \leq \delta$.

## Proof of the Performance Difference Lemma

**Step 1 — Telescoping with $V^\pi$.**

Note $V^\pi(s_0)$ averaged over $s_0 \sim \rho$ equals $J(\pi)$:
$$J(\pi) \;=\; \mathbb{E}_{s_0 \sim \rho}\!\bigl[V^\pi(s_0)\bigr].$$

Also, for any trajectory $\tau$:
$$V^\pi(s_0) = -\sum_{t=0}^\infty \gamma^t\bigl(\gamma V^\pi(s_{t+1}) - V^\pi(s_t)\bigr) \tag{telescoping, $V^\pi(s_\infty)=0$}.$$
(Verify: $-\gamma V^\pi(s_1) + V^\pi(s_0) - \gamma^2 V^\pi(s_2) + \gamma V^\pi(s_1) - \dots = V^\pi(s_0)$.)

**Step 2 — Add and subtract $V^\pi$ in $J(\pi')$.**

$$J(\pi') = \mathbb{E}_{\tau \sim \pi'}\!\biggl[\sum_t \gamma^t r_t\biggr]
       = \mathbb{E}_{\tau \sim \pi'}\!\biggl[V^\pi(s_0) + \sum_t \gamma^t\bigl(r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)\bigr)\biggr].$$

(Adding the telescoping identity, which equals $V^\pi(s_0)$, then subtracting it back as the first term.)

**Step 3 — Identify the TD error.**

Let $\delta^\pi_t := r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$. Then:
$$J(\pi') = \mathbb{E}_{s_0 \sim \rho}[V^\pi(s_0)] + \mathbb{E}_{\tau \sim \pi'}\!\biggl[\sum_t \gamma^t \delta^\pi_t\biggr] \;=\; J(\pi) + \mathbb{E}_{\tau \sim \pi'}\!\biggl[\sum_t \gamma^t \delta^\pi_t\biggr].$$

**Step 4 — Identify the advantage.**

Take expectation of $\delta^\pi_t$ conditional on $(s_t, a_t)$:
$$\mathbb{E}[\delta^\pi_t \mid s_t, a_t] = R(s_t, a_t) + \gamma\,\mathbb{E}_{s'}[V^\pi(s')] - V^\pi(s_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t).$$

Substituting:
$$J(\pi') - J(\pi) = \mathbb{E}_{\tau \sim \pi'}\!\biggl[\sum_t \gamma^t A^\pi(s_t, a_t)\biggr] \;=\; \frac{1}{1-\gamma}\,\mathbb{E}_{s \sim d^{\pi'}_\gamma,\,a \sim \pi'}\!\bigl[A^\pi(s, a)\bigr]. \qquad \square$$

## From PDL to the surrogate

The exact PDL depends on $d^{\pi'}_\gamma$ — the **new** policy's state distribution — which we can't sample without running $\pi'$.

**Surrogate definition:** replace $d^{\pi'}_\gamma$ with $d^\pi_\gamma$ (the **current** distribution we can sample):
$$L_\pi(\pi') := J(\pi) + \frac{1}{1-\gamma}\,\mathbb{E}_{s \sim d^\pi_\gamma,\, a \sim \pi'(\cdot|s)}\!\bigl[A^\pi(s, a)\bigr].$$

**Convert to importance sampling** on $\pi$ (so we can compute it from on-policy samples):
$$\mathbb{E}_{a \sim \pi'(\cdot|s)}[A^\pi(s, a)] = \mathbb{E}_{a \sim \pi(\cdot|s)}\!\left[\frac{\pi'(a|s)}{\pi(a|s)} A^\pi(s, a)\right].$$

So:
$$L_\pi(\pi') = J(\pi) + \frac{1}{1-\gamma}\,\mathbb{E}_{s \sim d^\pi_\gamma,\, a \sim \pi(\cdot|s)}\!\left[\frac{\pi'(a|s)}{\pi(a|s)} A^\pi(s, a)\right].$$

**First-order accuracy:** $L_\pi(\pi)|_{\pi'=\pi} = J(\pi)$ and $\nabla_{\pi'} L_\pi(\pi')|_{\pi'=\pi} = \nabla_{\pi'} J(\pi')|_{\pi'=\pi}$ (both equal the policy-gradient expression). So $L$ is a Taylor approximation around $\pi$ that ignores $d^{\pi'} - d^\pi$.

## Approximation bound and trust region

**Why the surrogate is good when policies are close:** the only error in $L_\pi(\pi') - (J(\pi') - J(\pi))$ comes from $d^{\pi'} \neq d^\pi$. By a coupling argument:
$$\|d^{\pi'}_\gamma - d^\pi_\gamma\|_1 \leq \frac{2\gamma}{1-\gamma}\, \max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 \leq \frac{2\gamma}{1-\gamma}\,\sqrt{2 D_{\text{KL}}^{\max}}.$$
(The last step is [[Pinskers Inequality]].)

Bounding the difference in expectations by total variation times $\|A^\pi\|_\infty$:
$$|J(\pi') - L_\pi(\pi')| \leq \frac{1}{1-\gamma}\cdot \varepsilon \cdot \|d^{\pi'}-d^\pi\|_1 \leq \frac{4\varepsilon\gamma}{(1-\gamma)^2} D_{\text{KL}}^{\max}(\pi, \pi').$$

**Monotonic improvement (lower bound):**
$$J(\pi') \;\geq\; L_\pi(\pi') - \frac{4\varepsilon\gamma}{(1-\gamma)^2}\,D_{\text{KL}}^{\max}(\pi, \pi').$$

If we **maximize** the RHS over $\pi'$, then $J(\pi') \geq J(\pi)$ (since the bound is tight at $\pi' = \pi$). This gives a *guaranteed* improvement step.

## TRPO algorithm

In practice, the penalty version above is replaced with a **constrained** version (easier to tune):

$$\boxed{\;\max_{\theta'} \;\mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta}\!\left[\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}\, A^{\pi_\theta}(s,a)\right] \quad \text{s.t.} \quad \mathbb{E}_{s \sim d^{\pi_\theta}}\bigl[\mathrm{KL}(\pi_\theta(\cdot|s) \,\|\, \pi_{\theta'}(\cdot|s))\bigr] \leq \delta.\;}$$

Solved by linearizing the objective (policy gradient) and quadratic-approximating the KL constraint (Fisher information matrix). Update is a natural gradient step with backtracking line search.

**PPO** simplifies by replacing the constraint with a clipped surrogate:
$$L^{\text{CLIP}}(\theta') = \mathbb{E}\!\Bigl[\min\bigl(r_t(\theta') A_t,\; \mathrm{clip}(r_t(\theta'), 1-\epsilon, 1+\epsilon) A_t\bigr)\Bigr], \quad r_t(\theta') = \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}.$$

## Intuition / what to remember

- **PDL is the load-bearing identity:** $J(\pi') - J(\pi) = \tfrac{1}{1-\gamma}\mathbb{E}_{d^{\pi'}}[A^\pi]$. Memorize it.
- **The proof is "telescoping with $V^\pi$".** Take $\tau \sim \pi'$, insert $V^\pi$ across rollouts, identify advantages.
- **The surrogate cheats by using $d^\pi$ instead of $d^{\pi'}$.** This is valid first-order; the second-order error is controlled by KL.
- **Importance sampling ratio $\pi'/\pi$ appears naturally** when you switch from $a \sim \pi'$ to $a \sim \pi$ in the expectation.
- **Trust region = the radius within which the surrogate is trustworthy.** Outside it, the second-order KL error dominates.
- **Why PPO works:** clipping the IS ratio is a cheap proxy for the KL constraint — no second-order math.

## See also

- [[Policy Gradient Theorem]] — what $L_\pi$ reduces to at first order.
- [[KL Divergence]] / [[Pinskers Inequality]] — the trust-region geometry.
- [[Follow the Regularized Leader]] — TRPO is structurally FTRL with KL regularizer on policies.
