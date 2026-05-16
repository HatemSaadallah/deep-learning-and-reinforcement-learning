# Regret Decomposition Lemma

**Tags:** #bandits #lemma #foundational

The cleanest identity in [[Stochastic Bandits]] analysis.

## Statement

Let $N_a(t) = \sum_{j=1}^t \mathbb{1}[a_j = a]$ be the number of pulls of arm $a$ in the first $t$ rounds. Then
$$R^T_\mathcal{D} = \sum_{a \in [K]} \Delta_a \cdot \mathbb{E}[N_a(T)].$$

## Proof sketch

Start from $R^T = T\mu^* - \mathbb{E}[\sum_t r_t]$, use $\sum_a \mathbb{1}\{a_t = a\} = 1$ for every $t$:
$$\sum_t r_t = \sum_t \sum_a r_t \mathbb{1}\{a_t = a\}.$$
Tower property: $\mathbb{E}[r_t \mid a_t] = \mu_{a_t}$. Then
$$R^T = \mathbb{E}\Big[\sum_t \sum_a \mathbb{1}\{a_t = a\}(\mu^* - \mu_a)\Big] = \sum_a \Delta_a \mathbb{E}[N_a(T)]. \;\;\square$$

## Why it's load-bearing

Reduces regret analysis to *counting bad pulls*. Every bandit algorithm is essentially trying to keep $\mathbb{E}[N_a(T)]$ small for sub-optimal arms. UCB, ETC, Thompson Sampling — they all attack this quantity.
