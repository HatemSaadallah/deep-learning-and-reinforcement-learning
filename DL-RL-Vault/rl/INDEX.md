# Reinforcement Learning — folder index

Hub for the RL side of the vault. Mirror of [[dl/INDEX|the DL folder index]].

## Foundations (`rl/foundations/`)

Mathematical tools and meta-principles used throughout RL theory.

**Concentration / information theory**
- [[Hoeffding Inequality]] — i.i.d. bounded variables
- [[Martingales and Bernstein]] — dependent sequences + variance-aware bounds
- [[Clean Event]] — the high-probability event UCB analyses live on
- [[KL Divergence]] — used in lower bounds, TRPO, RLHF
- [[Pinskers Inequality]] — KL → total variation
- [[Yaos Principle]] — randomised → deterministic lower-bound reduction

**Convex analysis**
- [[Convexity]] — convex sets/functions, strong convexity, exp-concavity hierarchy
- [[Bregman Divergence]] — generalised distance for mirror descent
- [[Three-point Equality]] — the algebraic identity behind MD analysis
- [[Euclidean Mirror Descent Lemma]] — one-step inequality for OGD
- [[Exp-Concavity]] — curvature condition giving $\log T$ regret
- [[Smoothness]] — $L$-smoothness, gradient descent lemma

**Meta-principles**
- [[Optimism Principle]] — the design recipe behind UCB / UCRL / LinUCB
- [[Nash Equilibrium]] — solution concept connecting regret minimisation to games

## Online learning (`rl/online-learning/`)

- [[Online Learning]] *(setting hub)*
- [[Regret]] — the performance measure
- [[Online Convex Optimization]] *(framework hub)*
- [[Follow the Leader]] / [[Follow the Regularized Leader]] / [[Follow the Perturbed Leader]]
- [[Hedge - Multiplicative Weights]]
- [[Optimistic FTRL]] — $O(1/T)$ rates in self-play
- [[Online Ridge Regression]] / [[Online Logistic Regression]]
- [[Adversarial Expert Problem]]

## Bandits (`rl/bandits/`)

**Stochastic**
- [[Stochastic Bandits]] *(setting hub)*
- [[Sub-optimality Gap]] / [[Regret Decomposition Lemma]]
- [[Epsilon-Greedy]] / [[Explore-Then-Commit]] / [[UCB1]] / [[Thompson Sampling]]
- [[Best-Arm Identification]] / [[Bandit Lower Bound]]

**Adversarial**
- [[Adversarial Bandits]] *(setting hub)* / [[EXP3]]

**Contextual / linear**
- [[Contextual Bandit]] / [[LinUCB]]

## MDP planning (`rl/mdp/`)

Tabular, classical (everything assumes finite state/action spaces and known or simulator-accessible dynamics).

- [[Markov Decision Process]] *(setting hub)*
- [[Bellman Equations]]
- [[Value Iteration]] / [[Policy Iteration]]
- [[Q-learning]] — model-free incremental VI
- [[Exploration in RL]] — UCB-VI, Q-learning with UCB

## Deep RL (`rl/deep-rl/`)

Function-approximation / neural-net RL. Where modern RL practice lives.

- [[Deep Policy Evaluation]] — MC, TD, k-step TD, stop-gradient
- [[REINFORCE and Actor-Critic]] — A2C / A3C family
- [[PPO and GRPO]] — trust-region policy optimisation
- [[RLHF and DPO]] — RL for LLM alignment

## Exam-prep proofs (`rl/proofs/`)

→ [[INDEX|Proofs index]] (full list with topic mapping, study guide, conventions)

| Proof | One-line trick |
|---|---|
| [[OGD Regret Bound]] | Telescope $\\|x_t - x^*\\|^2$ |
| [[ETC Regret Analysis]] | Hoeffding + union bound + optimise $T_0, \epsilon$ |
| [[UCB Analysis]] | Decompose $\{a_t = a\}$ into 3 events / clean event |
| [[FTRL Stability and Regret]] | Be-the-leader + drift |
| [[Mirror Descent Analysis]] | Three-point Bregman identity |
| [[Lipschitz Contextual Bandits Proof]] | $\epsilon$-net + Cauchy-Schwarz |
| [[Regret and Equilibria]] | Add P1/P2 inequalities → minimax sandwich |
| [[Markov Policies Suffice]] | Backward induction via Bellman |
| [[VI Generative Setting]] | Fixed-point reduction + Hoeffding on $\langle P, V^*\rangle$ |
| [[Policy Gradient Theorem]] | Log-derivative + causality |
| [[TRPO Surrogate Objective]] | PDL + IS + KL trust region |

## How the two sides connect

| RL topic | Related DL topic |
|---|---|
| [[Policy Gradient Theorem]] / [[REINFORCE and Actor-Critic]] | [[Optimisation]] (Adam(W) underneath) |
| [[RLHF and DPO]] | [[Transformer]] (the underlying LLM) |
| [[Optimisation]] of the value network | [[Deep Policy Evaluation]] |
| [[Regret and Equilibria]] (zero-sum game theory) | [[GANs]] (also a zero-sum game) |
| [[KL Divergence]] (TRPO trust region) | [[VAEs]] (KL in ELBO), [[RLHF and DPO]] (KL to reference policy) |

See [[dl/INDEX]] for the DL side.
