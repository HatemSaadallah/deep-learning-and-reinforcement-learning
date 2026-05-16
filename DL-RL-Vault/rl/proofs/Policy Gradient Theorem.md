# Proof — Policy Gradient Theorem

**Tags:** #proof #policy-gradient #foundational
**Topic:** Closed-form expression for $\nabla_\theta J(\theta)$ under a parameterized policy.

## Statement

Let $\pi_\theta(a | s)$ be a differentiable policy parameterization. Define
$$J(\theta) := \mathbb{E}_{\tau \sim P_\theta}\!\left[\sum_{t=0}^\infty \gamma^t r_t\right].$$

**Theorem (Sutton et al. 1999):**
$$\boxed{\;\nabla_\theta J(\theta) \;=\; \frac{1}{1-\gamma}\,\mathbb{E}_{s \sim d^{\pi_\theta}_\gamma,\, a \sim \pi_\theta(\cdot|s)}\!\Bigl[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a)\Bigr].\;}$$

Equivalently (REINFORCE form, trajectory-based):
$$\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim P_\theta}\!\left[\sum_{t=0}^\infty \gamma^t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t\right]$$
where $G_t = \sum_{t' \geq t} \gamma^{t'-t} r_{t'}$.

## Setup

- Trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$.
- Trajectory distribution: $P_\theta(\tau) = \rho(s_0)\,\prod_t P(s_{t+1}|s_t, a_t)\,\pi_\theta(a_t|s_t)$.
- Discounted return: $G(\tau) = \sum_{t \geq 0} \gamma^t r_t$.
- Discounted state distribution: $d^{\pi}_\gamma(s) := (1-\gamma)\sum_{t\geq 0} \gamma^t \Pr_\pi[s_t = s]$ (normalized so $\sum_s d^\pi_\gamma(s) = 1$).

## Proof (log-derivative trick)

**Step 1 — Differentiate under the expectation.**

$$\nabla_\theta J(\theta) = \nabla_\theta \int P_\theta(\tau)\, G(\tau)\, d\tau \;=\; \int \nabla_\theta P_\theta(\tau)\, G(\tau)\, d\tau.$$

**Step 2 — Log-derivative ("score function") trick.**

For any $P_\theta > 0$: $\nabla_\theta P_\theta = P_\theta \cdot \nabla_\theta \log P_\theta$.
$$\nabla_\theta J(\theta) = \int P_\theta(\tau)\, \nabla_\theta \log P_\theta(\tau)\, G(\tau)\, d\tau \;=\; \mathbb{E}_\tau\!\bigl[\nabla_\theta \log P_\theta(\tau) \cdot G(\tau)\bigr].$$

**Step 3 — Expand $\log P_\theta(\tau)$.**

$$\log P_\theta(\tau) = \log \rho(s_0) + \sum_t \log P(s_{t+1}|s_t, a_t) + \sum_t \log \pi_\theta(a_t|s_t).$$

The initial-state distribution and the transitions **do not depend on $\theta$**, so their gradients vanish:
$$\nabla_\theta \log P_\theta(\tau) \;=\; \sum_{t \geq 0} \nabla_\theta \log \pi_\theta(a_t | s_t).$$

Substituting:
$$\nabla_\theta J(\theta) \;=\; \mathbb{E}_\tau\!\Biggl[\Bigl(\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\Bigr) \cdot G(\tau)\Biggr].$$

**Step 4 — Causality / "rewards-to-go".**

The gradient $\nabla \log \pi(a_t | s_t)$ multiplies the *entire* return $G(\tau) = \sum_{t' \geq 0} \gamma^{t'} r_{t'}$. But $a_t$ cannot affect past rewards $r_{t'}$ for $t' < t$ (causality). Concretely, for $t' < t$:
$$\mathbb{E}\!\bigl[\nabla \log \pi_\theta(a_t|s_t) \cdot \gamma^{t'} r_{t'}\bigr] = \gamma^{t'}\,\mathbb{E}\bigl[r_{t'}\bigr] \cdot \underbrace{\mathbb{E}\!\bigl[\nabla \log \pi_\theta(a_t|s_t) \mid s_t\bigr]}_{= 0}.$$

(The score function has zero conditional mean: $\mathbb{E}_{a \sim \pi_\theta}[\nabla \log \pi_\theta(a|s)] = \nabla\!\sum_a \pi_\theta(a|s) = \nabla 1 = 0$.)

So past rewards drop out and we may replace $G(\tau)$ with $\sum_{t' \geq t} \gamma^{t'} r_{t'}$, or equivalently $\gamma^t G_t$:
$$\nabla J = \mathbb{E}_\tau\!\biggl[\sum_t \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t\biggr].$$

**Step 5 — Replace $G_t$ with $Q^{\pi_\theta}(s_t, a_t)$.**

By definition of $Q^\pi$: $\mathbb{E}[G_t \mid s_t, a_t, \pi_\theta] = Q^{\pi_\theta}(s_t, a_t)$. Take the tower property:
$$\nabla J = \mathbb{E}_\tau\!\biggl[\sum_t \gamma^t \nabla \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\biggr].$$

Rewrite using the discounted state distribution:
$$\nabla J = \sum_t \gamma^t\, \mathbb{E}_{s_t \sim \pi_\theta,\, a_t \sim \pi_\theta(\cdot|s_t)}\!\bigl[\nabla \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\bigr] \;=\; \frac{1}{1-\gamma}\, \mathbb{E}_{s \sim d^{\pi_\theta}_\gamma,\, a \sim \pi_\theta(\cdot|s)}\!\bigl[\nabla \log \pi_\theta \cdot Q^{\pi_\theta}\bigr]. \quad \square$$

## Baseline / advantage trick

For any state-only baseline $b: \mathcal{S} \to \mathbb{R}$:
$$\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\!\bigl[\nabla\log \pi_\theta(a|s) \cdot b(s)\bigr] = b(s) \cdot \mathbb{E}_a[\nabla\log \pi_\theta(a|s)] = b(s) \cdot 0 = 0.$$

So we can subtract $b(s)$ from $Q^\pi(s,a)$ without changing the gradient:
$$\nabla J = \frac{1}{1-\gamma}\,\mathbb{E}\bigl[\nabla\log \pi_\theta(a|s) \cdot (Q^{\pi_\theta}(s,a) - b(s))\bigr].$$

Choosing $b(s) = V^{\pi_\theta}(s)$ gives the **advantage** $A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$. The advantage form has much **lower variance** in Monte-Carlo estimation (since on average $Q^\pi \approx V^\pi$).

## Intuition / what to remember

- **Two key tricks:** (a) log-derivative / score function, (b) causality (zero-mean score). Everything else is bookkeeping.
- **Transitions and initial state don't depend on $\theta$ → their gradients drop out.** This is why policy gradient doesn't need a model.
- **Score function has zero conditional mean** → you can subtract any state-only baseline.
- **REINFORCE (Williams 1992)** uses Monte-Carlo $G_t$; **Actor-Critic** uses an estimated $\hat Q$ or $\hat A$ via TD; **PPO/TRPO** use clipped/constrained surrogate objectives (see [[TRPO Surrogate Objective]]).

## See also

- [[Q-learning]] — value-based alternative.
- [[TRPO Surrogate Objective]] — what to optimize when raw gradients are noisy.
