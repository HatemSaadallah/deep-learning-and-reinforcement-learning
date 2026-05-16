# Contextual Bandit

**Tags:** #bandits #framework

A [[Stochastic Bandits|bandit]] where each round the environment reveals a **context** $x_t$ before the agent picks an arm. The optimal arm depends on the context.

## Setup

At round $t$:
1. Environment reveals context $x_t \in \mathcal{X}$.
2. Agent picks arm $a_t \in \mathcal{A}$.
3. Reward $r_t \sim \mathcal{D}(x_t, a_t)$, mean $\mu(x_t, a_t)$.

Best policy: $a^*(x) = \arg\max_a \mu(x, a)$.

[[Regret]]:
$$R^T = \sum_{t=1}^T \mu(x_t, a^*(x_t)) - \mathbb{E}\!\left[\sum_{t=1}^T r_t\right].$$

## Why not run $|\mathcal{X}|$ independent bandits?

Contexts are usually continuous or astronomically large (users, webpages, feature vectors). You'd almost never see the same context twice → no learning.

**Fix: assume structure across contexts.** Most common:

$$\mu(x, a) = \langle \theta_a, x \rangle$$

— **linear contextual bandit**. Now it's a regression problem with bandit feedback: one $\theta_a$ per arm, shared across all contexts.

## Position in the bandit/RL hierarchy

| | State? | Action affects future state? |
|---|---|---|
| [[Stochastic Bandits]] | no | no |
| **Contextual bandit** | yes ($x_t$, i.i.d.) | **no** |
| MDP / RL ([[Q-learning]]) | yes ($s_t$) | yes |

The critical distinction from full RL: **actions don't influence the next context.** No credit assignment, no bootstrapping needed. This is why contextual bandits are tractable and beloved in industry.

## Algorithms

- [[LinUCB]] — UCB1 generalized to linear payoffs. Regret $\widetilde{O}(\sqrt{dT})$.
- **Contextual Thompson Sampling** — Gaussian posterior over $\theta_a$, sample-and-act-greedy.
- **Neural contextual bandits** — replace linear with NN, do TS via dropout / ensembles. What real recommender systems run.
- **Epoch-greedy / explore-then-exploit** — contextual analogue of [[Explore-Then-Commit]].

## Canonical application

**Yahoo News personalized article selection** (Li et al. 2010):
- $x_t$: user features
- arms: candidate articles
- reward: click (0/1)
- Why bandit? Logged data only contains feedback for shown articles.
- Why contextual? Different users prefer different content.

This template — context → action → partial feedback — describes essentially every modern personalized recommender, ad ranker, and A/B-testing platform.

## Connection to supervised learning

A contextual bandit is supervised classification under two changes:
1. **You** pick the label $a_t$ — no oracle labels for unchosen actions.
2. Evaluation is online, against regret rather than accuracy.

This connection underlies **off-policy evaluation** (importance sampling), **counterfactual learning**, and the bandit-feedback view of RLHF.

## References

- Li, Chu, Langford, Schapire. *A contextual-bandit approach to personalized news article recommendation.* WWW 2010.
- Lattimore & Szepesvári, *Bandit Algorithms* (2020), Ch. 18–22.
