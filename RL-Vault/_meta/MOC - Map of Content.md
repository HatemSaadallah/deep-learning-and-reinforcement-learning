# Map of Content (MOC)

Top-level entry point into the vault. Use the graph view (`Ctrl+G`) to see the live structure.

## Foundations

- [[Online Learning]]
- [[Regret]]
- [[Convexity]]
- [[Hoeffding Inequality]]
- [[Clean Event]]
- [[KL Divergence]]
- [[Pinskers Inequality]]
- [[Yaos Principle]]
- [[Bregman Divergence]]
- [[Three-point Equality]]
- [[Euclidean Mirror Descent Lemma]]
- [[Exp-Concavity]]
- [[Optimism Principle]]

## Online learning algorithms

- [[Online Convex Optimization]] *(framework hub)*
- [[Follow the Leader]]
- [[Follow the Regularized Leader]]
- [[Follow the Perturbed Leader]]
- [[Hedge - Multiplicative Weights]]
- [[Online Ridge Regression]] — recursive least squares
- [[Online Logistic Regression]]

## Bandits

- [[Stochastic Bandits]] *(main hub)*
- [[Sub-optimality Gap]]
- [[Regret Decomposition Lemma]]
- [[Epsilon-Greedy]]
- [[Explore-Then-Commit]]
- [[UCB1]]
- [[Thompson Sampling]]
- [[Best-Arm Identification]]
- [[Bandit Lower Bound]]
- [[Contextual Bandit]] — bridge to RL
- [[LinUCB]]

## Exam proofs

→ [[INDEX|Proofs index]] (the 10–12 listed by Celli as potential exam open questions).
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

## Course timeline (Bocconi RL 20876)

- ✅ **Part 2** (2026-03-27, Celli): Stochastic bandits — ETC, $\Omega(\sqrt{KT})$ lower bound. See [[Stochastic Bandits]].
- ⏳ **Part 3** (next): UCB — closes the gap.

## To write (stubs — make a note when you study these)

- [[Online Gradient Descent]]
- [[Mirror Descent]]
- [[Adversarial Bandits]]
- [[Self-Play and Game-Theoretic RL]]
- [[PPO]]
- [[TRPO]]
- [[Q-learning]] — explained in chat 2026-05-16, not yet written
- [[SARSA]]
- UCB1 (Upper Confidence Bound) — implemented in `code/01_bandits/`
- Thompson Sampling — implemented in `code/01_bandits/`
- KL-UCB
- Contextual Thompson Sampling

## Conventions

- One concept per file. Filename = concept name.
- Liberal `[[wikilinks]]` — broken links are fine, they're TODOs.
- Tag with `#topic-area` for filtering.
- Math in `$ ... $` (inline) or `$$ ... $$` (block).
