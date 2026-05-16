# Markov Decision Process (MDP)

**Tags:** #mdp #framework #foundational
**Source:** RL course — intro to RL / "Beyond a single state" (2026-04-20)

The mathematical model underlying [[Q-learning|reinforcement learning]]. Generalizes [[Stochastic Bandits|bandits]] by adding **state**: actions trigger transitions to a new state, so the agent's choices influence the future.

## Definition

An MDP is a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, r, \mathbb{P}, T)$ (episodic form):

- $\mathcal{S}$: set of states.
- $\mathcal{A}$: set of actions.
- $r_t(s, a) \in [0, 1]$: reward at step $t$ for action $a$ in state $s$ (deterministic WLOG).
- $\mathbb{P}_t(s' \mid s, a)$: transition probability — **Markov property** ($s'$ depends only on current $(s, a)$, not history).
- $T$: episode length (horizon).

### Interaction protocol

```
for each episode k = 1, ..., K:
    learner decides policy π
    initial state s_1 ~ P_1
    for t = 1, ..., T:
        learner picks a_t ~ π_t(·|s_t)
        environment transitions to s_{t+1} ~ P_t(·|s_t, a_t)
        learner observes s_{t+1} and reward r_t
```

## Infinite-horizon discounted setting

Alternative formulation: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, r, \mathbb{P}, \gamma)$ with **discount factor** $\gamma \in (0, 1)$. Reward at step $t$ counts as $\gamma^t r(s_t, a_t)$.

**Effective horizon** $\approx 1/(1-\gamma)$: rewards beyond $T \gtrsim \log(1/\epsilon)/\log(1/\gamma) \approx \log(1/\epsilon)/(1-\gamma)$ contribute at most $\epsilon$ to the total. So discount essentially **truncates** the problem at this effective horizon.

The episodic and discounted formulations are largely interchangeable — proofs adapt.

## Environment knowledge

How much of $\mathbb{P}, r$ does the learner know?

| Setting | What's accessible | Algorithms |
|---|---|---|
| **Known** | $\mathbb{P}, r$ explicit → pure **planning** | [[Value Iteration]], [[Policy Iteration]] |
| **Simulator / generative model** | query any $(s, a)$ → sample $r, s'$ | sample-based [[VI Generative Setting\|VI]], DP with empirical model |
| **Unknown — online** | interact from initial state, no resets | RL: [[Q-learning]], policy gradient, UCRL/UCB-VI |
| **Unknown — offline** | fixed dataset of trajectories | offline / batch RL: BCQ, CQL, conservative methods |

## Policies

A **policy** specifies a behavior for every state. Two key axes:

| | Deterministic | Stochastic |
|---|---|---|
| **History-dependent** | $\pi_t(s_1, a_1, \dots, s_t) \in \mathcal{A}$ | $\pi_t(\cdot \mid s_1, a_1, \dots, s_t) \in \Delta(\mathcal{A})$ |
| **Markov** | $\pi_t(s_t) \in \mathcal{A}$ | $\pi_t(\cdot \mid s_t) \in \Delta(\mathcal{A})$ |

**Key fact:** [[Markov Policies Suffice|Markov policies suffice]] — there always exists an optimal policy that is **Markov and deterministic**. So we can search just $\Pi_M$, the set of Markov policies, without loss of generality.

## Value functions

For a policy $\pi$ and state $s$ at time $t$, the **value function**:
$$V_t^\pi(s) \;:=\; \mathbb{E}_\pi\!\left[\sum_{t'=t}^T r_{t'}(s_{t'}, a_{t'}) \;\Big|\; s_t = s\right].$$

For a state-action pair $(s, a)$ at time $t$, the **Q-function**:
$$Q_t^\pi(s, a) \;:=\; \mathbb{E}_\pi\!\left[\sum_{t'=t}^T r_{t'}(s_{t'}, a_{t'}) \;\Big|\; s_t = s,\; a_t = a\right].$$

Relationship: $V_t^\pi(s) = \sum_a Q_t^\pi(s, a)\, \pi_t(a \mid s) = \mathbb{E}_{a \sim \pi_t(\cdot|s)}[Q_t^\pi(s, a)]$.

## Goal

Find $\pi^* = \arg\max_\pi V_1^\pi(s_1)$ (fixed start) or $\arg\max_\pi \mathbb{E}_{s_1 \sim \mu}[V_1^\pi(s_1)]$ (stochastic start).

The optimal value functions:
$$V_t^*(s) = \max_\pi V_t^\pi(s), \qquad Q_t^*(s, a) = \max_\pi Q_t^\pi(s, a).$$

Both satisfy the [[Bellman Equations]] (evaluation form for $V^\pi, Q^\pi$; optimality form for $V^*, Q^*$).

## Tabular notation

When $|\mathcal{S}|, |\mathcal{A}|$ are finite, view $V_t^\pi$ as a vector in $\mathbb{R}^{|\mathcal{S}|}$, $Q_t^\pi$ as a vector in $\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$. The transition matrix $\mathbb{P}_t \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}| \times |\mathcal{S}|}$ acts on values: $(\mathbb{P}_t V)(s, a) = \mathbb{E}_{s' \sim \mathbb{P}_t(\cdot|s,a)}[V(s')]$.

## Bandit ↔ MDP correspondence

| Bandit | MDP |
|---|---|
| Arms $a$ | Actions $a$ at each state |
| Reward $\mu(a)$ | Reward $r(s, a)$ |
| — | Transition $\mathbb{P}(\cdot \mid s, a)$ |
| Cumulative reward | Discounted / undiscounted return |
| Regret vs best arm | Regret vs best policy |

A bandit is an MDP with $|\mathcal{S}| = 1$ and trivial transitions. RL inherits and generalizes everything from bandit theory.

## See also

- [[Bellman Equations]] — the recursive structure of values.
- [[Value Iteration]] / [[Policy Iteration]] — the planning algorithms.
- [[Q-learning]] — sample-based learning of $Q^*$.
- [[Markov Policies Suffice]] — the proof.
- [[VI Generative Setting]] — sample complexity with a simulator.
- [[Policy Gradient Theorem]] — gradient of value w.r.t. policy parameters.
