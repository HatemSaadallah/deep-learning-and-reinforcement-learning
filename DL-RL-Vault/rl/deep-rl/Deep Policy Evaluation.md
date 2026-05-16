# Deep Policy Evaluation (MC, TD, k-step TD)

**Tags:** #mdp #algorithm #deep-rl
**Source:** the "Deep Policy Evaluation" lecture, 2026-05-11.

How to estimate $V^\pi(s)$ when the state space is too large for tabular methods → approximate with a **value network** $V_\phi(s)$ trained on sample trajectories.

## The supervised regression view

If we knew $V^\pi(s)$ exactly, we'd just fit a network by minimizing
$$\mathcal{L}(\phi) = \mathbb{E}_{s \sim \rho}\!\left[\tfrac{1}{2}\bigl(V_\phi(s) - V^\pi(s)\bigr)^2\right].$$
Gradient: $\nabla_\phi \mathcal{L} = \mathbb{E}[(V_\phi(s) - V^\pi(s)) \nabla_\phi V_\phi(s)]$.

**Problem:** we don't know $V^\pi(s)$. Need to **estimate** it from samples — that's where MC and TD come in.

## Monte Carlo (MC) policy evaluation

**Idea:** roll out trajectories with $\pi$, use realized cumulative reward as an estimate of $V^\pi$.
$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{T-1} \gamma^t r_t \;\big|\; s_0 = s\right] \;\approx\; \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T^{(i)}-1} \gamma^t r_t^{(i)}.$$

**Unbiased estimator** of $V^\pi(s)$, but only computable after the **entire trajectory** terminates.

### MC + SGD training of $V_\phi$

Sample trajectory $\tau$. Replace true $V^\pi(s_0)$ with the realized return $G_0(\tau) = \sum_t \gamma^t r_t$. Stochastic gradient:
$$g = \bigl(V_\phi(s_0) - G_0(\tau)\bigr) \nabla_\phi V_\phi(s_0).$$
**Unbiased** estimate of $\nabla \mathcal{L}(\phi)$.

**Downside:** must complete the episode before updating. Wasteful when $V_\phi$ is already accurate in late stages.

## Temporal Difference (TD) learning

**Key Bellman identity:** $V^\pi(s) = \mathbb{E}[r_0 + \gamma V^\pi(s_1) \mid s_0 = s]$.

**Idea:** instead of waiting for full return, use the **one-step bootstrap** $r_0 + \gamma V^\pi(s_1)$ — but $V^\pi$ is unknown, so substitute the current estimate $V_\phi$:
$$r_0 + \gamma V_\phi(s_1) \;\approx\; V^\pi(s).$$

TD gradient (treating the right side as the regression target):
$$g = \bigl(V_\phi(s_0) - r_0 - \gamma V_\phi(s_1)\bigr) \nabla_\phi V_\phi(s_0).$$

### Stop-gradient (critical implementation detail)

Naively backpropagating through $V_\phi(s_1)$ is **wrong** — it's the *target*, not part of the loss to optimize. In PyTorch use `.detach()`:
```python
target = r + gamma * V_phi(s1).detach()
loss = 0.5 * (V_phi(s0) - target).pow(2)
loss.backward()
```
This treats $V_\phi(s_1)$ as a fixed constant. Without `.detach()`, the gradient drives $V_\phi(s_1)$ down to reduce the squared error — destabilizing training.

### TD bias

$g$ is a **biased** estimator of $\nabla \mathcal{L}$ (because $V_\phi \neq V^\pi$ initially). But the bias decreases as $V_\phi$ improves — a virtuous cycle.

## k-step TD: interpolating MC and TD

$$\hat V_t^{(k)} \;=\; \sum_{t'=0}^{k \wedge T - 1} \gamma^{t'} r_{t+t'} + \gamma^{k \wedge T}\, V_\phi(s_{t + (k \wedge T)})$$
where $k \wedge T = \min(k, T)$ (handles early termination).

- $k = 1$: pure TD.
- $k = \infty$: pure MC.
- Intermediate $k$ (e.g. $k = 5$): partial bootstrap, partial roll-out. Trades off bias and variance.

## MC vs TD trade-off

| | **MC** | **TD** |
|---|---|---|
| Unbiased? | ✓ | ✗ (until $V_\phi$ converges) |
| Update frequency | end of episode | every step |
| Variance | high (full trajectory noise) | low (one-step noise) |
| Convergence | always works (no contraction needed) | can diverge with function approximation |
| Computational cost per update | high | low |
| Early-training behavior | safe but slow | possibly unstable |

In practice, **k-step TD with $k \approx 5$** is a sweet spot — used in A2C/A3C, DQN with n-step returns, PPO.

## Connection to RL vocabulary

- TD learning is the **value-evaluation half** of [[Q-learning]] (Q-learning bootstraps $Q$ on the next state's max; TD bootstraps $V$ on the next state's actual or sampled action).
- **GAE** (Generalized Advantage Estimation) is a weighted combination of k-step TD estimates with exponential decay $\lambda$ — used in TRPO/PPO.

## See also

- [[Bellman Equations]] — the identity TD exploits.
- [[Q-learning]] — value-based sample algorithm.
- [[REINFORCE and Actor-Critic]] — uses $V_\phi$ as a critic for variance reduction in policy gradients.
- [[TRPO Surrogate Objective]] — uses GAE built from k-step TD.
