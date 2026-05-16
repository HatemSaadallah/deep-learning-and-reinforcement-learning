# Bellman Equations

**Tags:** #mdp #foundational #lemma

The recursive structure of value functions in an [[Markov Decision Process|MDP]]. Underlies all of dynamic programming for RL.

## Bellman equation (policy evaluation form)

For any Markov policy $\pi$:
$$\boxed{\;\begin{aligned}
V_t^\pi(s) &= \sum_{a \in \mathcal{A}} Q_t^\pi(s, a)\, \pi_t(a \mid s) \\
Q_t^\pi(s, a) &= r_t(s, a) + \mathbb{E}_{s' \sim \mathbb{P}_t(\cdot|s,a)}\bigl[V_{t+1}^\pi(s')\bigr] \;=\; r_t(s, a) + (\mathbb{P}_t V_{t+1}^\pi)(s, a)
\end{aligned}\;}$$

with boundary $V_{T+1}^\pi(s) = 0$.

**Derivation:** by Markov property of the transition kernel, conditioning on $(s_t, a_t) = (s, a)$ makes the future independent of history. So $\mathbb{E}[\sum_{t'\geq t} r_{t'} | s_t = s, a_t = a]$ decomposes as $r_t(s,a) + \mathbb{E}_{s'}[V_{t+1}^\pi(s')]$.

## Bellman optimality equation

For the **optimal** value functions:
$$\boxed{\;\begin{aligned}
V_t^*(s) &= \max_{a \in \mathcal{A}} Q_t^*(s, a) \\
Q_t^*(s, a) &= r_t(s, a) + (\mathbb{P}_t V_{t+1}^*)(s, a)
\end{aligned}\;}$$

with boundary $V_{T+1}^*(s) = 0$. The optimal **greedy policy** is
$$\pi_t^*(s) = \arg\max_a Q_t^*(s, a).$$

The change vs. the evaluation form: replace $\sum_a \pi(a|s) \cdot$ with $\max_a$. This **single change** is what turns "evaluate $\pi$" into "compute the optimal $\pi$" — the heart of [[Value Iteration]].

## Why it's load-bearing

- **Reduces $\Theta((SA)^H)$ trajectory sum to $\Theta(SAH)$ DP.** Naively, $V_1^\pi(s_1) = \sum_\tau P^\pi(\tau|s_1) R(\tau)$ requires summing over all trajectories — exponentially many. Bellman exploits the Markov property to reduce to a layered DP.
- **Foundation of every RL algorithm.** [[Q-learning]] is sample-based Bellman *optimality* iteration. [[Policy Gradient Theorem]] uses Bellman to relate $\nabla V$ to $\nabla \log \pi \cdot Q$. [[TRPO Surrogate Objective]]'s performance difference lemma is "telescoping with $V^\pi$" — exactly a Bellman manipulation.

## Discounted infinite-horizon version

Stationary case (drop the time index): for $\gamma \in (0, 1)$,
$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}\bigl[r(s, a) + \gamma \mathbb{E}_{s'}[V^\pi(s')]\bigr].$$

The corresponding Bellman operator $\mathcal{T}^\pi V(s) = \sum_a \pi(a|s)[r(s,a) + \gamma \mathbb{E}_{s'} V(s')]$ is a **$\gamma$-contraction** in $\|\cdot\|_\infty$ — hence iterating it converges to the unique fixed point $V^\pi$ (this is the underpinning of policy evaluation by fixed-point iteration, and of [[VI Generative Setting|sample-based VI]]).

The Bellman optimality operator $\mathcal{T} V(s) = \max_a [r(s,a) + \gamma \mathbb{E}_{s'} V(s')]$ is also a $\gamma$-contraction. Its fixed point is $V^*$.

## See also

- [[Markov Decision Process]] — the setting.
- [[Value Iteration]] — applies Bellman optimality iteratively.
- [[Policy Iteration]] — alternates Bellman evaluation + greedy improvement.
- [[Markov Policies Suffice]] — uses Bellman in the proof.
- [[VI Generative Setting]] — sample complexity of finding the fixed point.
