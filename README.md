# Deep Learning and Reinforcement Learning

Personal study workspace combining a knowledge graph and runnable algorithm implementations.

- **Obsidian vault** (`DL-RL-Vault/`) — ~80 cross-linked concept notes covering deep learning and reinforcement learning, including exam-prep proofs and mock exams.
- **Working code** (`code/`) — algorithm implementations alongside the notes.

## Structure

```
.
├── DL-RL-Vault/                 # Obsidian vault — open this folder as a vault
│   ├── _meta/                   #   MOC (start here)
│   ├── dl/                      #   Deep learning (14 hub notes)
│   ├── rl/                      #   Reinforcement learning
│   │   ├── foundations/         #     Math (Hoeffding, KL, Bregman, ...)
│   │   ├── online-learning/     #     FTL, FTRL, OGD, Hedge, OMD
│   │   ├── bandits/             #     Stochastic, adversarial, contextual
│   │   ├── mdp/                 #     Tabular planning (VI, PI, Q-learning)
│   │   ├── deep-rl/             #     REINFORCE, A2C, PPO, GRPO, RLHF, DPO
│   │   └── proofs/              #     Exam-prep proofs
│   └── mocks/                   #   Practice exams + master cheat sheet
│
├── code/
│   └── 01_bandits/              # Stage 1: ETC/UCB/Thompson on Bernoulli arms
│
└── .venv/                       # Python 3.13 + numpy + matplotlib (gitignored)
```

## Quick start

```bash
# Bandit experiments (reproduces the regret-curve plots)
source .venv/bin/activate
cd code/01_bandits
python plot_regret.py            # writes regret_curves.png, regret_loglog.png
```

The code reproduces the regret behaviors from the stochastic-bandits lecture: linear regret for Uniform / FTL, $\widetilde{O}(T^{2/3})$ for ETC, $\widetilde{O}(\sqrt{KT})$ for UCB1 and Thompson Sampling. Thompson dominates empirically on the test instance.

## Using the Obsidian vault

1. Open Obsidian → "Open folder as vault" → select `DL-RL-Vault/`.
2. Hit `Ctrl+G` for the graph view.
3. Start at `_meta/MOC - Map of Content.md`.
4. Deep-dive into one half via the folder indexes: [`dl/INDEX.md`](DL-RL-Vault/dl/INDEX.md), [`rl/INDEX.md`](DL-RL-Vault/rl/INDEX.md).

## Exam-prep proofs

Located at `DL-RL-Vault/rl/proofs/`. Start at `INDEX.md`. Each proof is a self-contained note with statement, setup, full derivation, and a "what to remember" line on the load-bearing steps.

| Topic | File |
|---|---|
| OGD regret $O(\sqrt{T})$ | `OGD Regret Bound.md` |
| ETC analysis $\widetilde{O}(T^{2/3})$ | `ETC Regret Analysis.md` |
| UCB instance-dep + indep | `UCB Analysis.md` |
| FTRL stability + regret | `FTRL Stability and Regret.md` |
| Mirror descent + simplex OMD | `Mirror Descent Analysis.md` |
| Lipschitz contextual bandits | `Lipschitz Contextual Bandits Proof.md` |
| Regret ↔ Nash (0-sum games) | `Regret and Equilibria.md` |
| Markov policies suffice | `Markov Policies Suffice.md` |
| VI generative setting | `VI Generative Setting.md` |
| Policy gradient theorem | `Policy Gradient Theorem.md` |
| TRPO surrogate objective | `TRPO Surrogate Objective.md` |

## Mock exams

Located at `DL-RL-Vault/mocks/`. Two practice exams plus a 3-page master cheat sheet covering the whole syllabus.

| Mock | Coverage |
|---|---|
| `2024-05 RL Final Exam.md` | 16 questions, mixed RL + DL |
| `2025-03 DL Mock.md` | 13 questions, pure DL |
| `Cheat Sheet.md` | Full-course concept map (handwritten, navigation links into the vault) |

Answer keys are in collapsible sections — try the questions before peeking.

## Roadmap (code stages)

- [x] **Stage 1:** stochastic bandits — ETC, UCB1, Thompson Sampling, ε-greedy, regret plots
- [ ] **Stage 2:** tabular Q-learning on FrozenLake / Taxi (planned)
- [ ] **Stage 3:** DQN on CartPole (planned)
- [ ] **Stage 4:** PPO on LunarLander / Atari (planned)
- [ ] Adversarial bandits (EXP3) — vault note exists, code TODO
- [ ] Contextual bandit (LinUCB) — vault note exists, code TODO

## Dependencies

The `.venv/` is gitignored. To recreate:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

For future stages: add `torch`, `gymnasium[classic-control,box2d,atari]`.

## References

- Lattimore & Szepesvári, *Bandit Algorithms* (2020) — the bandit half.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2018) — RL foundations.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016) — DL fundamentals.
- Hazan, *Introduction to Online Convex Optimization* (2022) — OCO / FTRL.

Topics covered in the vault: bandits (stochastic, contextual, adversarial), online learning (FTL/FTRL/MD), zero-sum games, MDPs, value iteration, policy gradients, TRPO/PPO, RLHF/DPO, plus deep-learning fundamentals (optimisation, init, regularisation, transformers, GNNs, VAEs, GANs, diffusion).
