# REINFORCE and Actor-Critic Methods

**Tags:** #mdp #algorithm #policy-gradient
**Source:** Williams 1992 (REINFORCE), Mnih et al. 2016 (A3C / A2C). Celli "Policy Optimization" lecture, 2026-05-11.

The **policy-based** family of RL algorithms: parameterize the policy $\pi_\theta$ directly, optimize $J(\theta) = \mathbb{E}[\sum_t \gamma^t r_t]$ via gradient ascent on $\nabla J(\theta)$ from the [[Policy Gradient Theorem]].

## The policy gradient (recap)

From [[Policy Gradient Theorem|the theorem]]:
$$\nabla J(\theta) \;=\; \mathbb{E}_\tau\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \gamma^t G_t(\tau)\right]$$
where $G_t(\tau) = \sum_{t' \geq t} \gamma^{t'-t} r_{t'}$ is the return from time $t$.

**General form** (any unbiased $\hat Q_t$ and state-baseline $b(s)$):
$$\nabla J(\theta) \;=\; \mathbb{E}_\tau\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \gamma^t (\hat Q_t - b(s_t))\right].$$

The choice of $\hat Q_t$ and $b(s_t)$ trades off bias vs. variance. The algorithms below differ in this choice.

## REINFORCE (Williams 1992)

Simplest choice: $\hat Q_t = G_t(\tau)$ (Monte-Carlo return), $b = 0$.

**Algorithm:**
```
Init θ_0
For each episode k:
    sample trajectory τ ~ π_{θ_k}
    compute G_t for each t
    θ_{k+1} ← θ_k + η_k · Σ_t γ^t ∇ log π_θ(a_t | s_t) · G_t
```

**Property:** unbiased gradient, but **extreme variance** — especially when:
- Rewards are sparse (pass/fail, e.g. LLM reasoning evaluation).
- No per-action credit assignment (all $a_t$ get blamed/credited for the same $G$).
- Trajectories are long (noise grows linearly in $T$).

Everything after REINFORCE is **variance reduction**.

## Variance reduction 1: causality (remove past rewards)

Action $a_t$ cannot influence rewards before time $t$. So replace $G_0(\tau)$ (full trajectory return) with $G_t$ (future return only):
$$\nabla J = \mathbb{E}\!\left[\sum_t \nabla\log\pi_\theta(a_t|s_t)\, \gamma^t G_t\right].$$
**Proof:** the cross-terms $\mathbb{E}[\nabla\log\pi(a_t|s_t) \cdot r_{t'}] = 0$ for $t' < t$ by causality + zero-mean score function. See [[Policy Gradient Theorem]].

## Variance reduction 2: baseline

For any state-only function $b(s)$:
$$\mathbb{E}_{a \sim \pi(\cdot|s)}[\nabla \log \pi(a|s) \cdot b(s)] = 0.$$
So subtract $b(s_t)$ without bias:
$$\nabla J = \mathbb{E}\!\left[\sum_t \nabla\log\pi_\theta(a_t|s_t)\, \gamma^t \bigl(G_t - b(s_t)\bigr)\right].$$

Optimal baseline (minimum variance) is approximately $b(s_t) = V^{\pi_\theta}(s_t)$. With this choice, the quantity $G_t - V^{\pi}(s_t)$ approximates the **advantage** $A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$.

## The Advantage view

$$A^\pi(s, a) := Q^\pi(s, a) - V^\pi(s).$$

- $A^\pi(s, a) > 0$: $a$ is **better than average** under $\pi$ at $s$.
- $A^\pi(s, a) < 0$: $a$ is **worse than average**.
- For optimal $\pi^*$: $A^{\pi^*}(s, a) \leq 0$ everywhere.

The PG update with advantage:
$$\nabla J = \mathbb{E}\!\left[\sum_t \nabla\log\pi_\theta(a_t|s_t)\, \gamma^t\, A^{\pi_\theta}(s_t, a_t)\right].$$
**Sign tells you the direction of policy update:** good actions get reinforced, bad ones suppressed.

## Actor-Critic methods

If we knew $Q^\pi$ exactly, we could plug it in. We don't, so we **learn** a value approximator (the **critic**) alongside the policy (the **actor**).

| Component | What it does | How it's trained |
|---|---|---|
| **Actor** $\pi_\theta$ | takes actions | policy gradient update with critic-based advantage estimate |
| **Critic** $V_\phi$ | evaluates actions | [[Deep Policy Evaluation\|TD or MC regression]] toward $V^{\pi_\theta}$ |

## Why use $\hat Q_t - V_\phi(s_t)$ as the advantage estimate?

- **$\hat Q_t$ via k-step TD:** $\hat Q_t = \sum_{t'=0}^{k-1} \gamma^{t'} r_{t+t'} + \gamma^k V_\phi(s_{t+k})$.
- Approximates $Q^{\pi_\theta}(s_t, a_t)$.
- Even when $V_\phi$ is initially wrong, the **observed rewards** $r_t, \dots, r_{t+k-1}$ provide unbiased information about action quality.
- Subtracting $V_\phi(s_t)$ centers the estimate around 0.

Setting $k = 1$ gives the 1-step TD advantage; $k = \infty$ recovers MC (REINFORCE with baseline). Intermediate $k$ trades bias vs. variance.

## A2C — Advantage Actor-Critic (Mnih et al. 2016)

**Synchronous** variant of A3C. Both actor and critic are neural networks, often sharing trunks.

```
While not converged:
    Reset state if terminal
    For k steps (k ≈ 5):
        a_t ~ π_θ(·|s_t); receive r_t, s_{t+1}
        if s_{t+1} terminal: break
    
    Bootstrap: Q̂ = V_φ(s_{T}) if non-terminal else 0
    For i = T-1, T-2, ..., t_start:
        Q̂ ← r_i + γ Q̂           # k-step return
        g_θ -= ∇log π_θ(a_i|s_i) · (Q̂ - V_φ(s_i))     # actor
        g_φ += ∇ (1/2)([Q̂] - V_φ(s_i))²              # critic ([·] = stop-grad)
    
    Update θ, φ via optimizer (SGD/Adam)
```

Key implementation details:
- **Entropy bonus** added to actor loss: $-\beta \, H[\pi_\theta(\cdot|s)]$ → encourages exploration.
- **Stop-gradient** on the bootstrap target ($[\hat Q]$) — see [[Deep Policy Evaluation]].
- **Shared trunk** between actor and critic networks.

## A3C — Asynchronous Advantage Actor-Critic

Multiple parallel actor-critic learners on different CPU cores, each with its own environment instance. Asynchronous gradient updates to a shared parameter server. **Key insight:** parallelism replaces experience replay for breaking correlations between consecutive samples.

In simulators where sampling is cheap (Atari, MuJoCo), A2C/A3C are extremely sample-inefficient but **wall-clock fast**. In real-world settings (robotics, RLHF), the sample efficiency matters → more sophisticated methods like [[TRPO Surrogate Objective|TRPO]], PPO, SAC are preferred.

## Family overview

```
REINFORCE  (MC, no baseline)
   │
   ├─→ REINFORCE with baseline  (variance reduced)
   │       │
   │       ├─→ Actor-Critic  (learned critic V_φ)
   │       │       │
   │       │       ├─→ A2C / A3C  (k-step advantage, NN critic, parallel)
   │       │       │       │
   │       │       │       ├─→ TRPO  (trust region constraint on policy update)
   │       │       │       └─→ PPO  (clipped surrogate, simpler trust region)
   │       │       └─→ SAC  (entropy-regularized, off-policy)
   │       └─→ GPO / DDPG  (deterministic policy gradient)
```

## See also

- [[Policy Gradient Theorem]] — the foundational gradient.
- [[Deep Policy Evaluation]] — how the critic $V_\phi$ is trained.
- [[TRPO Surrogate Objective]] — the trust-region successor.
- [[Q-learning]] — value-based alternative (no policy network).

## References

- Williams, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, ML 1992 (REINFORCE).
- Mnih et al., *Asynchronous methods for deep reinforcement learning*, ICML 2016 (A3C). [arXiv:1602.01783](https://arxiv.org/abs/1602.01783).
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Ch. 13.
