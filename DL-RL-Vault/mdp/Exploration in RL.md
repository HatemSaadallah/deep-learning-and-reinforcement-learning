# Exploration in RL (Online MDP)

**Tags:** #mdp #framework #algorithm
**Source:** lecture 12, 2026-04-30.

The **online RL** setting: $\mathbb{P}, r$ are unknown, no generative model — the learner can only interact from an initial state $s_1$ and reset at episode end. The challenge is to **efficiently visit informative states** despite never having seen most of the MDP.

## The protocol

For episode $k = 1, 2, \dots$:
1. Reset to $s_1$.
2. For $h = 1, \dots, H$: agent picks $a_h$, environment transitions to $s_{h+1}$ and gives $r_h$.

Goal: minimize **regret over $K$ episodes**:
$$R_K = K \cdot V_1^*(s_1) \;-\; \sum_{k=1}^K V_1^{\pi_k}(s_1).$$

Equivalently, average over $K$ episodes converges to $V^*$.

## Why random exploration fails

**Combinatorial lock MDP** (slide 3):

Imagine an $H$-step MDP with two states $s^0$ (failure), $s^1$ (success). At each step $h$ there is exactly one "key action" $a_h^*$ that keeps you in $s^1$ — every other action sends you to $s^0$. Reward = 0 except a single $+1$ for reaching $s^1$ at step $H$.

- Optimal policy: $\pi^*(s^1) = a_h^*$ for each step. $V^*(s_1) = 1$.
- **Random exploration:** probability of playing the full $H$-key sequence is $(1/|\mathcal{A}|)^H$.
- To find $\epsilon$-optimal policy with $\epsilon$-greedy, need $\Omega(|\mathcal{A}|^H)$ episodes. **Exponential in horizon.**

→ Random exploration is fundamentally inadequate. We need to **systematically visit under-explored states**.

## Solution: optimism (again)

Same [[Optimism Principle|optimism principle]] as in bandits — act as if the best plausible MDP were true. Two main algorithms:

### UCB-VI (Azar et al. 2017)

Model-based. Maintain empirical model $(\hat{\mathbb{P}}, \hat{r})$ + **Hoeffding-style exploration bonus** on each $Q$ update:
$$Q_h(s, a) \leftarrow \min\bigl(H,\; r_h(s, a) + (\hat{\mathbb{P}}_h V_{h+1})(s, a) + b(N_h(s, a))\bigr).$$
$$b(N) = \sqrt{\frac{H^2 \iota}{N}}, \quad \iota = \log\frac{SAK}{p}.$$

Then play greedy w.r.t. $Q$ in the current episode, observe transitions, update model.

**Algorithm structure** (per episode):
1. **Phase I (planning):** for $h = H, \dots, 1$, do a Bellman update with the bonus, building $V_h^k$.
2. **Phase II (execution):** play greedy $a_h = \arg\max_a Q_h(s_h, a)$ for the whole episode, collect data.

**Regret:** $R_K \leq \widetilde{O}\bigl(\sqrt{H^3 SAT}\bigr) + H^3 S^2 A \iota^3$ where $T = KH$. The minimax-optimal rate is $\widetilde{O}(\sqrt{H^3 SAT})$ — UCB-VI is tight.

### Q-learning with UCB exploration (Jin et al. 2018)

Model-free. Online Q-learning with an exploration bonus added to the TD target:
$$Q_h(s_h, a_h) \leftarrow (1 - \alpha_t)Q_h(s_h, a_h) + \alpha_t\bigl(r_h + V_{h+1}(s_{h+1}) + b_t\bigr), \quad b_t = \sqrt{\frac{H^3 \iota}{t}}.$$

**Regret:** $R_K \leq \widetilde{O}(\sqrt{H^4 SAT})$ — an extra $\sqrt{H}$ over UCB-VI, the same kind of price model-free pays in the generative setting.

The **first model-free RL algorithm with provable polynomial regret** — opened a major line of work on sample-efficient model-free RL.

## Sample complexity vs regret

Regret $R_K \leq \widetilde{O}(\sqrt{H^3 SAT})$ implies that average per-episode suboptimality is $V^* - \tfrac{1}{K}\sum V^{\pi_k} \leq \widetilde{O}(\sqrt{H^3 SA / K})$. Setting this $\leq \epsilon$:
$$K \gtrsim \widetilde{O}\!\left(\frac{H^3 SA}{\epsilon^2}\right), \quad\text{total samples }T = KH = \widetilde{O}(H^4 SA/\epsilon^2).$$

| Setting | Sample complexity | Algorithm |
|---|---|---|
| **Generative** | $\widetilde{O}(H^4 SA / \epsilon^2)$ | [[VI Generative Setting\|VI]] / [[Q-learning]] |
| **Online (exploration)** | $\widetilde{O}(H^4 SA / \epsilon^2) +$ lower-order | UCB-VI / Q-learning-UCB |

Online RL matches the generative-model rate in leading-order terms — but with lower-order additive terms reflecting the cost of self-collecting data.

## Optimism principle universality — RL extension

The optimism strategy works in MDPs because the [[Optimism Principle|two universality conditions]] still hold:
1. Visiting a state $s$ and taking $a$ gives feedback on $r(s, a)$ and $\mathbb{P}(\cdot \mid s, a)$.
2. No direct cross-action information leakage (within tabular settings).

In settings with **function approximation** (deep RL), condition 2 effectively fails — pulling one $(s, a)$ updates the neural net's predictions everywhere. This is part of why **exploration in deep RL is much harder** than in tabular RL.

## What's the proof structure for UCB-VI regret?

**Key lemma (optimism on event $\mathcal{G}$):** $Q_h^k(s, a) \geq Q_h^*(s, a)$ and $V_h^k(s) \geq V_h^*(s)$ w.h.p.

**Then:**
$$R_K = \sum_k [V_1^*(s_1) - V_1^{\pi_k}(s_1)] \leq \sum_k [V_1^k(s_1) - V_1^{\pi_k}(s_1)].$$

The sum on the right is then bounded via the [[Bellman Equations|Bellman equation]] + bonus telescoping + [[Martingales and Bernstein|Azuma-Hoeffding]] to control the on-policy errors (because the trajectory is adaptive, not i.i.d.).

## References

- Azar, Osband, Munos. *Minimax regret bounds for reinforcement learning*. ICML 2017.
- Jin, Allen-Zhu, Bubeck, Jordan. *Is Q-learning provably efficient?* NeurIPS 2018.

## See also

- [[Markov Decision Process]] — the setting.
- [[VI Generative Setting]] — easier sibling (simulator).
- [[Optimism Principle]] — the meta-recipe.
- [[Martingales and Bernstein]] — the concentration tools used in the analysis.
- [[Q-learning]] — base algorithm extended with UCB bonus here.
