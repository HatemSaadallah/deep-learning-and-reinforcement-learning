# Stochastic Bandits

**Tags:** #bandits #framework #foundational
**Source:** Bocconi RL course (Celli, Part 2, 2026-03-27)

Special case of [[Online Learning]] with: finite action set, **stochastic** environment, **bandit feedback**.

## Setup

- $K$ arms, each with reward distribution $\mathcal{D}_a$, mean $\mu_a$.
- At round $t$: choose $a_t$, receive $r_t \sim \mathcal{D}_{a_t}$.
- Algorithm = sequential sampling strategy $a_{t+1} = F_t(a_1, r_1, \dots, a_t, r_t)$.
- Goal: maximize $\mathbb{E}[\sum_{t=1}^T r_t]$.

## Key quantities

- **Optimal arm:** $a^* \in \arg\max_a \mu_a$.
- **Best mean:** $\mu^* = \max_a \mu_a$.
- **Sub-optimality gap:** $\Delta_a := \mu^* - \mu_a$. See [[Sub-optimality Gap]].
- **Pseudo-regret:** $R^T_\mathcal{D} := T\mu^* - \mathbb{E}[\sum_{t=1}^T r_t]$. See [[Regret]].

## Core lemma

[[Regret Decomposition Lemma]]: $R^T = \sum_a \Delta_a \, \mathbb{E}[N_a(T)]$.

→ Regret is exactly the cost of pulling sub-optimal arms × how often you pull them. This is the **exploration / exploitation trade-off** in one equation.

## Algorithms studied

| Algorithm | Regret | Notes |
|---|---|---|
| Uniform exploration | $\Theta(T)$ | pure exploration |
| [[Follow the Leader]] (greedy) | $\Omega(T)$ | unstable, locks onto wrong arm |
| [[Explore-Then-Commit]] | $\widetilde{O}(T^{2/3})$ | first sublinear |
| **UCB** *(next lecture)* | $\widetilde{O}(\sqrt{KT})$ | matches LB |

## Lower bound

$R^T \geq \Omega(\sqrt{KT})$ for **any** bandit algorithm — see [[Bandit Lower Bound]].

## Classic motivating examples

- **Clinical trials** (Thompson 1933): $K$ treatments, allocate to next patient based on prior responses.
- **Online recommendations** (Li et al. 2010): which ad/article/movie to show.

## References

- Lattimore & Szepesvári, *Bandit Algorithms*, 2020.
- Thompson, *Biometrika* 25, 1933.
