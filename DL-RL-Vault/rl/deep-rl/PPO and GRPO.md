# PPO and GRPO

**Tags:** #mdp #algorithm #policy-gradient #deep-rl
**Source:** Schulman et al. 2017 (PPO); Shao et al. 2024 (GRPO, DeepSeek-R1). the "PPO & Co" lecture, 2026-05-13.

Trust-region policy optimization algorithms that simplify [[TRPO Surrogate Objective|TRPO]] by avoiding second-order optimization. **PPO** is the de-facto deep RL algorithm; **GRPO** is its descendant used in modern LLM post-training (DeepSeek-R1).

## Background: TRPO algorithm

From the [[TRPO Surrogate Objective|surrogate objective]] derivation:
$$\max_{\theta'} \;\sum_i \sum_t \gamma^t \frac{\pi_{\theta'}(a_t^{(i)} | s_t^{(i)})}{\pi_{\theta_{\text{cur}}}(a_t^{(i)} | s_t^{(i)})}\, \hat A_t^{(i)} \quad\text{s.t.}\quad \max_s \mathrm{KL}(\pi_{\theta_{\text{next}}}(\cdot|s) \,\|\, \pi_{\theta_{\text{cur}}}(\cdot|s)) \leq \delta.$$
**Solved via approximate Newton + conjugate gradient** — needs Hessian-vector products, expensive.

## Generalized Advantage Estimation (GAE)

The advantage estimate $\hat A_t$ used by TRPO/PPO interpolates between [[Deep Policy Evaluation|MC and k-step TD]] via an exponentially weighted sum of TD residuals.

**TD residual:** $\delta_t := r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.

**GAE estimate** with hyperparameter $\lambda \in [0, 1]$:
$$\hat A_t^{\mathrm{GAE}(\gamma, \lambda)} \;:=\; \sum_{i=0}^{T-t-1} (\gamma \lambda)^i\, \delta_{t+i}.$$

- $\lambda \to 0$: trust critic heavily → 1-step TD, low variance, more bias.
- $\lambda \to 1$: similar to Monte Carlo → high variance, low bias. Setting $\gamma = 1$ recovers $\hat A_t = \sum_{t' \geq t} r_{t'} - V_\phi(s_t)$ exactly.

Standard practice: $\gamma = 0.99$, $\lambda = 0.95$ — trust critic but include long-horizon rewards.

## Proximal Policy Optimization (PPO)

**Idea:** return to first-order optimization, enforce the trust region **implicitly** via a clipped surrogate.

### Clipped surrogate objective

Define the importance-sampling ratio $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{cur}}}(a_t|s_t)$. The PPO clipped objective:
$$\boxed{\;C_\epsilon(r, \hat A) \;:=\; \min\bigl(r\, \hat A,\; \mathrm{clip}(r, 1-\epsilon, 1+\epsilon)\, \hat A\bigr).\;}$$

Equivalently, by sign of $\hat A$:
- If $\hat A \geq 0$: $C_\epsilon = \min(r, 1+\epsilon) \hat A$ — capped from above.
- If $\hat A < 0$: $C_\epsilon = \max(r, 1-\epsilon) \hat A$ — capped from above (less negative).

**Effect:** the loss has **zero gradient** when $r$ leaves $[1-\epsilon, 1+\epsilon]$ — so SGD won't push the policy further from $\pi_{\theta_{\text{cur}}}$ once outside the trust region. This is the trust region in disguise.

### PPO algorithm

```
While not converged:
    Sample N trajectories under π_{θ_cur}
    Compute Â_t via GAE using V_φ
    
    Solve via SGD/Adam with early stopping:
        max_{θ_next} Σ_i Σ_t γ^t · C_ε(π_{θ_next}/π_{θ_cur}, Â_t^{(i)})
    
    θ_cur ← θ_next
    
    Train critic V_φ to fit MC returns R̂_t:
        min_φ Σ_i Σ_t (1/2)(V_φ(s_t) - R̂_t)²
```

**Implementation notes:**
- $\epsilon \approx 0.2$ typical.
- "Solve" = a few epochs of SGD on the batch — clipping prevents over-fitting.
- Critic and actor often share trunk.
- Add entropy bonus to encourage exploration.

### Why PPO works

The clip + few-SGD-epochs combination keeps $\pi_\theta$ near $\pi_{\theta_{\text{cur}}}$ implicitly:
- Clipping creates a flat region where gradient = 0 outside $[1-\epsilon, 1+\epsilon]$.
- A few SGD steps within a single optimization round don't move far from initialization.
- Combined, this approximates the TRPO KL constraint without explicit constraint enforcement.

## Group Relative Policy Optimization (GRPO)

Shao et al. 2024, used in **DeepSeek-R1**. Removes the need for a critic $V_\phi$ — useful when:
- Critic is hard to train (sparse rewards, long trajectories — common in LLM reasoning).
- Reward only arrives at trajectory end.

### Setup

Consider an **undiscounted** MDP with reward only at terminal state $r^{(i)}$.

### Key idea

Replace the critic-based baseline $V_\phi(s_t)$ with a **group-relative normalization** across $N$ trajectories sampled from the same starting state:
$$\hat A_{(i)}^{\mathrm{GRPO}} \;=\; \frac{r^{(i)} - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r}) + \epsilon'}$$
where $\mathbf{r} = (r^{(1)}, \dots, r^{(N)})$. **Each trajectory's advantage is its z-score within the batch.**

### Algorithm

```
While not converged:
    Sample N trajectories from π_{θ_cur}, get rewards r^{(1)}, ..., r^{(N)}
    Compute group advantages Â^{GRPO}_{(i)} = (r^{(i)} - mean) / (std + ε')
    
    Solve (PPO clip with GRPO advantages):
        max_{θ_next} Σ_i Σ_t C_ε(π_{θ_next}(a_t|s_t)/π_{θ_cur}(a_t|s_t), Â^{GRPO}_{(i)})
    
    θ_cur ← θ_next
```

### Why GRPO works (and what can go wrong)

- **No critic = no critic bias.** Simpler training pipeline.
- **No critic = no per-step credit assignment.** Same advantage applied to every action in a trajectory — fine when reward only signals at the end.
- **Degenerate case:** if all $r^{(i)}$ are equal (e.g. all correct/incorrect), $\mathrm{std}(\mathbf{r}) = 0$, $\hat A = 0$, **no learning signal.** This is the main failure mode — need diverse samples.

### Why it's popular for LLM RL

LLM reasoning tasks have:
- Sparse, terminal-only reward (correct answer or not).
- Hard-to-train value function (one trajectory = one tokens-long sequence).
- Easy to sample many completions per prompt → easy to compute group statistics.

GRPO fits perfectly.

## Family comparison

| | Critic? | Trust region | Use case |
|---|---|---|---|
| [[REINFORCE and Actor-Critic\|A2C/A3C]] | yes | none | simulator-rich (Atari, MuJoCo) |
| [[TRPO Surrogate Objective\|TRPO]] | yes | hard KL constraint | safety-critical, control |
| **PPO** | yes | clipped surrogate | de-facto default, RLHF v0 |
| **GRPO** | no | clipped surrogate | LLM post-training (DeepSeek-R1) |

## See also

- [[Policy Gradient Theorem]] — the foundation.
- [[TRPO Surrogate Objective]] — what PPO simplifies.
- [[REINFORCE and Actor-Critic]] — A2C is PPO's predecessor.
- [[RLHF and DPO]] — where PPO and GRPO get used for LLMs.
- [[Deep Policy Evaluation]] — how the critic $V_\phi$ is trained.
