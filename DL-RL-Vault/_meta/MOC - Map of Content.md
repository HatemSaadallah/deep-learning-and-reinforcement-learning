# Map of Content (MOC)

Top-level entry point into the vault. Two halves: deep learning and reinforcement learning, each with its own folder index. Use the graph view (`Ctrl+G`) to see the live structure.

## Deep Learning — `dl/`

→ [[dl/INDEX|DL folder index]] (full breakdown)

- [[Neural Network Fundamentals]] — backprop, loss, MLE, UAT
- [[Optimisation]] — SGD, Adam(W), schedules, loss landscape
- [[Initialisation]] — Xavier/He, vanishing/exploding gradients
- [[Regularisation]] — L2, dropout, BN/LN, data aug, implicit
- [[Overparameterisation]] — double descent, NTK, scaling laws
- [[Transformer]] — self-attention backbone
- [[Transformers Biology]] — ESM, AlphaFold, DNA models
- [[GNNs]] — message-passing graph nets
- [[Unsupervised Learning]] — pretraining paradigms
- [[VAEs]] — variational autoencoders
- [[GANs]] — generative adversarial nets
- [[Diffusion Models]] — DDPM, score-based, latent diffusion
- [[Mixed DL Techniques]] — practical recipes

---

## Reinforcement Learning — `rl/`

→ [[rl/INDEX|RL folder index]] (full breakdown)

### Foundations (`rl/foundations/`)

- [[Hoeffding Inequality]] · [[Martingales and Bernstein]] · [[Clean Event]]
- [[KL Divergence]] · [[Pinskers Inequality]] · [[Yaos Principle]]
- [[Convexity]] · [[Bregman Divergence]] · [[Three-point Equality]]
- [[Euclidean Mirror Descent Lemma]] · [[Exp-Concavity]] · [[Smoothness]]
- [[Optimism Principle]] · [[Nash Equilibrium]]

### Online learning (`rl/online-learning/`)

- [[Online Learning]] · [[Regret]] · [[Online Convex Optimization]]
- [[Follow the Leader]] · [[Follow the Regularized Leader]] · [[Follow the Perturbed Leader]]
- [[Hedge - Multiplicative Weights]] · [[Optimistic FTRL]]
- [[Online Ridge Regression]] · [[Online Logistic Regression]] · [[Adversarial Expert Problem]]

### Bandits (`rl/bandits/`)

- **Stochastic:** [[Stochastic Bandits]] · [[Sub-optimality Gap]] · [[Regret Decomposition Lemma]] · [[Epsilon-Greedy]] · [[Explore-Then-Commit]] · [[UCB1]] · [[Thompson Sampling]] · [[Best-Arm Identification]] · [[Bandit Lower Bound]]
- **Adversarial:** [[Adversarial Bandits]] · [[EXP3]]
- **Contextual / linear:** [[Contextual Bandit]] · [[LinUCB]]

### MDP planning (`rl/mdp/`)

- [[Markov Decision Process]] · [[Bellman Equations]]
- [[Value Iteration]] · [[Policy Iteration]]
- [[Q-learning]] · [[Exploration in RL]]

### Deep RL (`rl/deep-rl/`)

- [[Deep Policy Evaluation]] · [[REINFORCE and Actor-Critic]] · [[PPO and GRPO]] · [[RLHF and DPO]]

### Exam-prep proofs (`rl/proofs/`)

→ [[rl/proofs/INDEX|Proofs index]]

[[OGD Regret Bound]] · [[ETC Regret Analysis]] · [[UCB Analysis]] · [[FTRL Stability and Regret]] · [[Mirror Descent Analysis]] · [[Lipschitz Contextual Bandits Proof]] · [[Regret and Equilibria]] · [[Markov Policies Suffice]] · [[VI Generative Setting]] · [[Policy Gradient Theorem]] · [[TRPO Surrogate Objective]]

---

## Mock exams — `mocks/`

→ [[mocks/INDEX|Mocks index]]
- [[2024-05 RL Final Exam]] — full RL final, 16 questions, 31 pts
- [[2025-03 DL Mock]] — DL-only mock, 13 questions, 31 pts
- [[Cheat Sheet]] — full-course concept map (handwritten reference)

## To write (stubs)

- [[Online Gradient Descent]] · [[Mirror Descent]] · [[Self-Play and Game-Theoretic RL]] · [[SARSA]] · KL-UCB · Contextual Thompson Sampling

## Conventions

- One concept per file. Filename = concept name.
- Liberal `[[wikilinks]]` — links resolve by note name (not path), so moves don't break them.
- Tag with `#topic-area` for filtering.
- Math in `$ ... $` (inline) or `$$ ... $$` (block).

## Folder layout

```
DL-RL-Vault/
├── _meta/          (MOC)
├── dl/             (14 deep-learning notes)
├── rl/
│   ├── foundations/    (math + meta-principles)
│   ├── online-learning/
│   ├── bandits/
│   ├── mdp/            (tabular planning)
│   ├── deep-rl/        (PPO, RLHF, actor-critic)
│   └── proofs/         (exam-prep)
└── mocks/          (practice exams + cheat sheet)
```
