# Q-learning

**Tags:** #mdp #algorithm #model-free
**Source:** Watkins 1989; Celli Lecture 11 (generative setting analysis).

A **model-free** RL algorithm that learns $Q^*$ directly from sample transitions, without estimating the transition model $\mathbb{P}$. Incremental version of [[Value Iteration]] — each transition gives one update to one entry of the $Q$ table.

## Update rule

After observing transition $(s, a, r, s')$ at iteration $t$:
$$\boxed{\;Q_h^t(s, a) \;\leftarrow\; (1 - \alpha_t)\,Q_h^{t-1}(s, a) \;+\; \alpha_t\,\bigl(r_h(s, a) + V_{h+1}^{t-1}(s')\bigr)\;}$$

where $V_h^{t-1}(s) = \max_a Q_h^{t-1}(s, a)$ and $\alpha_t \in [0, 1]$ is a **learning rate**.

Equivalent rewriting (TD-error form):
$$Q_h^t(s, a) = Q_h^{t-1}(s, a) + \alpha_t\,\underbrace{\bigl[r + V_{h+1}^{t-1}(s') - Q_h^{t-1}(s, a)\bigr]}_{\text{TD error}}.$$

## Interpretation as perturbed gradient descent

The [[Bellman Equations|Bellman optimality]] equation is
$$Q^*(s, a) = r(s, a) + \mathbb{E}_{s'}[V^*(s')].$$

Q-learning minimizes the squared Bellman residual $\ell(Q) = \tfrac{1}{2}(Q(s,a) - r(s,a) - \mathbb{E}_{s'}[V^*(s')])^2$ via gradient descent:
$$Q^{t+1}(s, a) = Q^t(s, a) - \alpha_t \nabla \ell(Q^t).$$

But $\mathbb{E}_{s'}$ is intractable (need $\mathbb{P}$). **Fix:** replace it with a single sample $s' \sim \mathbb{P}(\cdot|s,a)$. This is a **stochastic / perturbed gradient descent** step — the same trick that makes SGD work.

## Algorithm (generative setting)

**Input:** learning rates $\{\alpha_t\}$, $Q^0 = 0$.

**For** $t = 1, 2, \dots$:
- For each $(s, a, h)$: sample $s' \sim \mathbb{P}_h(\cdot | s, a)$.
- $Q_h^t(s, a) \leftarrow (1-\alpha_t) Q_h^{t-1}(s, a) + \alpha_t (r_h(s, a) + V_{h+1}^{t-1}(s'))$.
- $V_h^t(s) = \max_a Q_h^t(s, a)$.

## Choosing the learning rate

The update $Q_h^t$ is a convex combination of past samples with weights $\gamma_t^i = \alpha_i \prod_{j=i+1}^{t}(1 - \alpha_j)$:
$$Q_h^t(s, a) = r_h(s, a) + \sum_{i=1}^t \gamma_t^i\, V_{h+1}^{i-1}(s_i'), \qquad \sum_i \gamma_t^i = 1.$$

- $\alpha_t = 1/t \;\Longrightarrow\; \gamma_t^i = 1/t$: **uniform weights** = empirical mean = matches VI.
- $\alpha_t < 1/t$: favors **early** samples.
- $\alpha_t > 1/t$: favors **late** samples.

**Practical choice:** $\alpha_t = \tfrac{H + 1}{H + t}$. Slightly > $1/t$, so it favors later (more accurate) samples — early predictions are based on $V^0 = 0$ which is wildly inaccurate.

## Sample complexity (generative setting)

With $\alpha_t = (H+1)/(H+t)$, w.p. $\geq 1 - p$:
$$\left\|\tfrac{1}{t}\sum_{i=1}^t Q_1^i - Q_1^*\right\|_\infty \;\leq\; c\sqrt{\tfrac{H^5 \iota}{t}}, \quad \iota = \log(HSA/p).$$

To get $\epsilon$-optimality: $t \gtrsim H^5 \iota / \epsilon^2$ per $(s,a)$, total $tSAH \gtrsim H^6 SA \iota / \epsilon^2$.

Compare to model-based [[VI Generative Setting|VI]] sample complexity:

| | Total samples |
|---|---|
| **Model-based VI** | $\widetilde{O}(H^4 SA / \epsilon^2)$ (optimal) |
| **Q-learning** | $\widetilde{O}(H^6 SA / \epsilon^2)$ |

Q-learning pays an extra $H^2$ factor — the price of **non-uniform weighting** (early bad estimates contaminate the running average). For large state spaces this is offset by Q-learning's $O(SAH)$ space (vs VI's $O(S^2AH)$ for the empirical model).

## Why model-free vs model-based?

| | Model-based ([[Value Iteration]]) | Model-free (Q-learning) |
|---|---|---|
| Builds $\hat\mathbb{P}$? | Yes | No |
| Memory | $O(S^2 A H)$ | $O(SAH)$ |
| Sample complexity | $\widetilde{O}(H^4 SA/\epsilon^2)$ | $\widetilde{O}(H^6 SA/\epsilon^2)$ |
| Generalizes to function approximation | Harder | Natural (Deep Q-Networks) |

Q-learning is the foundation of **DQN** (Mnih et al. 2015): replace the tabular $Q$ with a neural net $Q_\theta$, use replay buffer + target network to stabilize the off-policy + bootstrapping + function-approximation **deadly triad**.

## Properties

1. **Off-policy.** Updates use $\max_{a'} Q(s', a')$ — best *next* action, not the one actually taken. So you can learn $Q^*$ while exploring with any behavior policy (e.g. $\epsilon$-greedy).
2. **Bootstrapping.** Updates an estimate using another estimate (the $V$ at the next state). Cheap per step but creates correlations.
3. **Asynchronous, online.** Each transition updates one $(s, a)$ entry. No need for sweeps over the full state space (unlike VI).

## Convergence (tabular)

In the tabular setting, Q-learning converges to $Q^*$ with probability 1 if:
- Every $(s, a)$ is visited infinitely often.
- Step sizes satisfy Robbins-Monro: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$.

The first condition is why **exploration** matters in Q-learning (typically: $\epsilon$-greedy behavior policy).

## See also

- [[Bellman Equations]] — Q-learning is sample-based Bellman optimality iteration.
- [[Value Iteration]] / [[VI Generative Setting]] — model-based alternative.
- [[Markov Decision Process]] — the setting.
- [[Policy Gradient Theorem]] — the policy-based alternative (no $Q$ estimate).
