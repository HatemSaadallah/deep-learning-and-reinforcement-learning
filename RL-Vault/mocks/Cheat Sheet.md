# Master Cheat Sheet (Bocconi RL 20876)

**Tags:** #mock #cheatsheet #reference
**Source:** Handwritten 3-page concept map (`6.pdf` in the mocks folder) covering the entire course.

This is a **navigation map** of the cheat sheet, not a transcription — for the actual formulas study the original or the linked vault notes. The cheat sheet's structure (which I've preserved here) is itself a useful study guide because it groups concepts by how they relate.

## Page 1 — Online learning + stochastic bandits

| Box on cheat sheet | Vault note |
|---|---|
| Batch / sequential learning, risk minimization | [[Online Learning]] |
| Sequential learning, linear/logistic regression, RLS | [[Online Ridge Regression]], [[Online Logistic Regression]] |
| Online convex optimization | [[Online Convex Optimization]] |
| Stochastic bandits + pseudo-regret + $N_a(t)$ | [[Stochastic Bandits]], [[Regret Decomposition Lemma]] |
| Online GD ($R_T \leq R^2/2\eta + \eta L^2 T/2 \to RL\sqrt{T}$) | [[OGD Regret Bound]] |
| Hoeffding inequality | [[Hoeffding Inequality]] |
| KL divergence + chain rule + Pinsker | [[KL Divergence]], [[Pinskers Inequality]] |
| Random coin example for LB | [[Bandit Lower Bound]] |
| Explore-Then-Commit ($T_0 = (T/K)^{2/3}(\log T)^{1/3} \to O(T^{2/3})$) | [[Explore-Then-Commit]] |
| Exploration only / exploitation only failures | [[Stochastic Bandits]] |
| Lower bound $\Omega(\sqrt{KT})$ + best-arm identification | [[Bandit Lower Bound]], [[Best-Arm Identification]] |
| $\epsilon$-greedy (LB $\epsilon\frac{K-1}{K}\Delta_{\min}T$) | [[Epsilon-Greedy]] |
| UCB upper / lower confidence bound | [[UCB1]], [[Clean Event]], [[UCB Analysis]] |
| Optimism violations → sublinear regret not guaranteed | [[Optimism Principle]] |
| Thompson sampling | [[Thompson Sampling]] |
| O / Ω notation reminder | — |

## Page 2 — Adversarial + OMD

| Box on cheat sheet | Vault note |
|---|---|
| Adversarial expert problem ($\Omega(\sqrt{T \log K})$ LB) | [[Adversarial Expert Problem]] |
| MWU / Hedge ($\eta = \sqrt{\log K / T} \to O(\sqrt{T \log K})$) | [[Hedge - Multiplicative Weights]] |
| Gradient descent on simplex (suboptimal $\sqrt{KT}$) | [[OGD Regret Bound]] |
| Follow-The-Leader instability | [[Follow the Leader]] |
| FTRL ($R_T \leq D/\eta + \eta \sum \|g_t\|^2 \to G\sqrt{TD}$) | [[Follow the Regularized Leader]], [[FTRL Stability and Regret]] |
| Strong convexity + dual norm + Hölder | [[Convexity]], [[Bregman Divergence]] |
| Adversarial bandits / EXP3 (importance-weighting) | [[Adversarial Bandits]], [[EXP3]] |
| Games, Nash equilibrium | [[Nash Equilibrium]] |
| Regret ↔ saddle-point gap ($\gamma \leq (R_X + R_Y)/T \to 0$) | [[Regret and Equilibria]] |
| Different regularizers (entropy / quadratic / log barrier) | [[Mirror Descent Analysis]] |
| Optimism (better than $\sqrt{T}$) | [[Optimistic FTRL]] |
| L-smooth definition + descent + convergence $D_g(x^*, x_0)/(\eta T)$ | [[Smoothness]] |
| Bregman divergence, three-point | [[Bregman Divergence]], [[Three-point Equality]] |
| Projected GD | [[OGD Regret Bound]] |
| Proximal step → online mirror descent | [[Mirror Descent Analysis]] |

## Page 3 — Contextual bandits + MDPs

| Box on cheat sheet | Vault note |
|---|---|
| Contextual bandits (best-response policy regret) | [[Contextual Bandit]] |
| Lipschitz CB (discretize → $O(T^{2/3}(LK \log T)^{1/3})$) | [[Lipschitz Contextual Bandits Proof]] |
| Stochastic linear bandits (confidence ellipsoid) | [[LinUCB]] |
| MDP definition (S, A, $r_t$, $\mathbb{P}_t$, T) | [[Markov Decision Process]] |
| Markov property + episodic | [[Markov Decision Process]] |
| Environment knowledge (known / simulator / unknown) | [[Markov Decision Process]] |
| Policies — history-dependent vs Markov | [[Markov Decision Process]] |
| V-value, Q-value | [[Markov Decision Process]] |
| Markov policies suffice ($V^\pi = V^\mu$) | [[Markov Policies Suffice]] |
| Tabular notation ($V^\pi \in \mathbb{R}^S$, $Q^\pi \in \mathbb{R}^{SA}$) | [[Markov Decision Process]] |
| Infinite-horizon discounted ($r_t = \gamma^t r_0$) | [[Markov Decision Process]] |
| Naive optimization → exponential, Bellman → polynomial | [[Bellman Equations]] |
| Bellman equation + optimality equation | [[Bellman Equations]] |
| Value Iteration / Policy Iteration algorithms | [[Value Iteration]], [[Policy Iteration]] |

## Topics on the cheat sheet — top-left navigation list

The left margin of page 1 lists the **full course topics**:
- Batch/Sequential Learning
- Stochastic Bandits
- Adversarial Multi-Armed Bandits
- Online Mirror Descent
- Contextual / Linear Bandits
- Applications
- MDP Planning
- Generative Models
- MDP Exploration
- Policy Gradient
- LLM Post-Training

Most of these have at least one dedicated vault note. Use this list as a checklist when revising.

## Where the cheat sheet is incomplete

The handwritten sheet covers the **theory** part of the course thoroughly through MDP planning + VI/PI, but the deep-RL part (Policy Gradient, A2C/PPO, RLHF) is left at "to-cover" level. For those topics, the recent vault additions cover everything:

- [[Q-learning]]
- [[Exploration in RL]] (UCB-VI, Q-learning-UCB, combinatorial lock)
- [[Deep Policy Evaluation]] (MC, TD, k-step TD)
- [[REINFORCE and Actor-Critic]] (A2C / A3C family)
- [[PPO and GRPO]]
- [[RLHF and DPO]]

## How to use this for the exam

1. **Read top-to-bottom, left-to-right** through all three pages once. This re-loads the structure of the course in your mind.
2. **For each box**, click through to the linked vault note if you can't recall the contents.
3. **Highlight any concept you can't recall the definition of** within 10 seconds — those are your weak topics.
4. **For each weak topic**: read the vault note + try to redo at least one proof from [[INDEX|the proofs folder]].

## Source

Original handwritten file: `~/Downloads/rl-mocks/6.pdf` (3 pages, captioned "DLR B Statements.xopp"). Same content also appears in `1.pdf`, `3.pdf`, `5.pdf`, `7.pdf` (duplicates of [[2024-05 RL Final Exam]]) and `2.pdf`, `4.pdf` (duplicates of [[2025-03 DL Mock]]).
