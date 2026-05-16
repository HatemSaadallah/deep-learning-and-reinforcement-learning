# Proof — Markov Policies Suffice for Finite-Horizon MDPs

**Tags:** #proof #mdp #foundational
**Topic:** Optimality of Markov (history-independent) policies.

## Statement

Consider a finite-horizon MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, H, \mu)$ with horizon $H$.

Let $\Pi^{\text{hist}}$ be the class of all (possibly randomized) **history-dependent** policies $\pi_h(a \mid s_1, a_1, r_1, \dots, s_h)$.
Let $\Pi^{\text{Mark}}$ be the class of **Markov, deterministic** policies $\pi_h: \mathcal{S} \to \mathcal{A}$.

**Theorem:**
$$\boxed{\;\max_{\pi \in \Pi^{\text{hist}}} J(\pi) \;=\; \max_{\pi \in \Pi^{\text{Mark}}} J(\pi).\;}$$

There exists a deterministic Markov policy that is globally optimal — no matter how complex the candidates allowed.

## Setup

- $J(\pi) = \mathbb{E}_\pi\!\left[\sum_{h=1}^H R(s_h, a_h)\right]$ — expected total reward, $s_1 \sim \mu$.
- $V^\pi_h(s)$, $Q^\pi_h(s, a)$ — value functions at stage $h$.
- $V^*_h(s) := \sup_{\pi \in \Pi^{\text{hist}}} V^\pi_h(s)$ — the optimal value across **all** policies (initially history-dependent).

## Proof (backward induction)

**Goal:** show by backward induction on $h = H, H-1, \dots, 1$ that

(I) $V^*_h(s)$ is well-defined and depends only on $s$ (not on the history that led to $s$).

(II) There exists a Markov deterministic policy achieving $V^*_h(s)$ from stage $h$ onward.

**Base case ($h = H$).** At the final stage, no future to worry about:
$$V^*_H(s) = \max_a R(s, a).$$
The maximizer $\pi^*_H(s) = \arg\max_a R(s, a)$ is Markov and deterministic. (I), (II) hold.

**Inductive step.** Assume (I), (II) hold for $h+1$. Show for $h$.

Consider any history-dependent policy $\pi$ and any history $\tau_h = (s_1, a_1, \dots, s_h)$ ending in $s_h$. Conditioning on $\tau_h$:
$$V^\pi_h(\tau_h) = \mathbb{E}_\pi\!\Bigl[R(s_h, a_h) + V^\pi_{h+1}(\tau_h, a_h, s_{h+1}) \;\Big|\; \tau_h\Bigr].$$

By the **Markov property of the transition kernel** ($s_{h+1}$ depends on $\tau_h$ only through $(s_h, a_h)$):
$$\mathbb{E}_\pi[V^\pi_{h+1}(\tau_h, a_h, s_{h+1}) \mid \tau_h, a_h] = \mathbb{E}_{s' \sim P(\cdot \mid s_h, a_h)}\bigl[V^\pi_{h+1}(\tau_h, a_h, s')\bigr].$$

Upper-bound using the IH (any $V^\pi_{h+1}$ value $\leq V^*_{h+1}$, which depends only on the state):
$$V^\pi_h(\tau_h) \;\leq\; \max_a \Bigl\{R(s_h, a) + \mathbb{E}_{s' \sim P(\cdot|s_h, a)}[V^*_{h+1}(s')]\Bigr\}.$$

The RHS depends **only on $s_h$** — not on the rest of $\tau_h$. Define:
$$V^*_h(s) := \max_a \Bigl[R(s, a) + \mathbb{E}_{s' \sim P(\cdot | s, a)}\, V^*_{h+1}(s')\Bigr], \qquad Q^*_h(s, a) := R(s, a) + \mathbb{E}_{s'}\, V^*_{h+1}(s'). \tag{Bellman}$$

So **every** history-dependent policy satisfies $V^\pi_h(\tau_h) \leq V^*_h(s_h)$. This establishes (I).

**Existence of optimal Markov policy:** Define $\pi^*_h(s) := \arg\max_a Q^*_h(s, a)$ — Markov and deterministic. By induction, the policy $(\pi^*_h, \pi^*_{h+1}, \dots, \pi^*_H)$ achieves $V^*_{h+1}(s)$ from stage $h+1$, and at stage $h$ it picks the maximizer. So:
$$V^{\pi^*}_h(s) = R(s, \pi^*_h(s)) + \mathbb{E}_{s'}\, V^*_{h+1}(s') = V^*_h(s).$$
The bound is achieved. (II) holds. $\square$

## Why this fails for POMDPs

The proof uses the Markov property of $P$: $s_{h+1}$ depends on history only through $(s_h, a_h)$. In **partially observable** MDPs, the agent observes only $o_h$ (not $s_h$), and the relevant Markov state is the **belief** $b_h \in \Delta(\mathcal{S})$, which depends on the entire observation-action history. Policies must depend on history (or equivalently, on the belief state).

## Intuition / what to remember

- **Bellman's principle of optimality:** "An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision."
- **The Markov property of $P$ is what carries through.** No matter what history brings you to state $s$, the future is the same conditional on $s$. So you can't gain anything by remembering it.
- **The optimal action at stage $h$ depends only on $s_h$** and on the optimal value function at $h+1$ (which by IH depends only on $s$).
- **Determinism comes for free.** Even allowing randomization, an argmax of $Q^*_h(s, \cdot)$ is optimal.
- **This is what justifies tabular [[Q-learning]] and value iteration** in finite-horizon MDPs.

## See also

- [[Q-learning]] — algorithm built on Bellman optimality.
- [[VI Generative Setting]] — sample complexity of value iteration.
