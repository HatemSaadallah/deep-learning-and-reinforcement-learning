# Map of Content (MOC)

Top-level entry point into the vault. Use the graph view (`Ctrl+G`) to see the live structure.

## Foundations

- [[Online Learning]]
- [[Regret]]
- [[Convexity]]
- [[Hoeffding Inequality]]
- [[Martingales and Bernstein]]
- [[Clean Event]]
- [[KL Divergence]]
- [[Pinskers Inequality]]
- [[Yaos Principle]]
- [[Bregman Divergence]]
- [[Three-point Equality]]
- [[Euclidean Mirror Descent Lemma]]
- [[Exp-Concavity]]
- [[Smoothness]]
- [[Optimism Principle]]
- [[Nash Equilibrium]]

## Online learning algorithms

- [[Online Convex Optimization]] *(framework hub)*
- [[Follow the Leader]]
- [[Follow the Regularized Leader]]
- [[Follow the Perturbed Leader]]
- [[Optimistic FTRL]] — $O(1/T)$ in self-play
- [[Hedge - Multiplicative Weights]]
- [[Online Ridge Regression]] — recursive least squares
- [[Online Logistic Regression]]
- [[Adversarial Expert Problem]]

## Bandits

- [[Stochastic Bandits]] *(stochastic hub)*
- [[Sub-optimality Gap]]
- [[Regret Decomposition Lemma]]
- [[Epsilon-Greedy]]
- [[Explore-Then-Commit]]
- [[UCB1]]
- [[Thompson Sampling]]
- [[Best-Arm Identification]]
- [[Bandit Lower Bound]]
- [[Adversarial Bandits]] *(adversarial hub)*
- [[Adversarial Expert Problem]] — full-info adversarial
- [[EXP3]]
- [[Contextual Bandit]] — bridge to RL
- [[LinUCB]]

## Exam proofs

→ [[INDEX|Proofs index]] (the 10–12 listed as potential exam open questions).
- [[OGD Regret Bound]]
- [[ETC Regret Analysis]]
- [[UCB Analysis]] — instance-dep and instance-indep
- [[FTRL Stability and Regret]]
- [[Mirror Descent Analysis]] — general + OMD on simplex
- [[Lipschitz Contextual Bandits Proof]]
- [[Regret and Equilibria]]
- [[Markov Policies Suffice]]
- [[VI Generative Setting]]
- [[Policy Gradient Theorem]]
- [[TRPO Surrogate Objective]]

## Mock exams

→ [[INDEX|Mocks index]]
- [[2024-05 RL Final Exam]] — full RL final, 16 questions, 31 pts
- [[2025-03 DL Mock]] — DL-only mock, 13 questions, 31 pts
- [[Cheat Sheet]] — full-course concept map (handwritten reference)

## Course timeline

- ✅ **Part 2** (2026-03-27): Stochastic bandits — ETC, $\Omega(\sqrt{KT})$ lower bound. See [[Stochastic Bandits]].
- ⏳ **Part 3** (next): UCB — closes the gap.

## MDPs and Planning

- [[Markov Decision Process]] *(setting hub)*
- [[Bellman Equations]]
- [[Value Iteration]]
- [[Policy Iteration]]
- [[Q-learning]] — model-free incremental VI
- [[Exploration in RL]] — UCB-VI + Q-learning-UCB

## Deep RL

- [[Deep Policy Evaluation]] — MC, TD, k-step TD, stop-gradient
- [[REINFORCE and Actor-Critic]] — A2C/A3C family
- [[PPO and GRPO]] — trust-region policy optimization
- [[RLHF and DPO]] — RL for LLM alignment

## To write (stubs — make a note when you study these)

- [[Online Gradient Descent]]
- [[Mirror Descent]]
- [[Self-Play and Game-Theoretic RL]]
- [[SARSA]]
- KL-UCB
- Contextual Thompson Sampling

## Conventions

- One concept per file. Filename = concept name.
- Liberal `[[wikilinks]]` — broken links are fine, they're TODOs.
- Tag with `#topic-area` for filtering.
- Math in `$ ... $` (inline) or `$$ ... $$` (block).
