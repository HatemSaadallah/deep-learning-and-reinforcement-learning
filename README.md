# Deep Learning and Reinforcement Learning

Personal study workspace for **reinforcement learning** and **deep learning**. Combines:
- An **Obsidian vault** (`DL-DL-RL-Vault/`) — concept notes, cross-linked into a knowledge graph.
- **Working code** (`code/`) — algorithm implementations matched to the lectures.
- **Exam-prep proofs** (`DL-RL-Vault/proofs/`) — full proofs for every theorem flagged as "may appear on the exam".

## Structure

```
.
├── DL-RL-Vault/                   # Obsidian vault
│   ├── _meta/                  # Map of content (start here)
│   ├── foundations/            # Hoeffding, KL, Pinsker, Yao, ...
│   ├── online-learning/        # FTL, FTRL, FTPL, Hedge, Regret, ...
│   ├── bandits/                # Stochastic + contextual bandits, ETC, UCB, LinUCB
│   └── proofs/                 # Exam-ready proofs (see proofs/INDEX.md)
│
├── code/
│   └── 01_bandits/             # ETC / UCB / Thompson on Bernoulli arms, regret plots
│
└── .venv/                      # Python 3.13 + numpy + matplotlib (not committed)
```

## Reproducing the bandit experiments

```bash
source .venv/bin/activate
cd code/01_bandits
python plot_regret.py            # writes regret_curves.png, regret_loglog.png
```

The code reproduces the regret behaviors from the lecture slides: linear regret for Uniform / FTL, $\widetilde{O}(T^{2/3})$ for ETC, $\widetilde{O}(\sqrt{KT})$ for UCB1 and Thompson Sampling.

## Using the Obsidian vault

1. Open Obsidian → "Open folder as vault" → select `DL-RL-Vault/`.
2. `Ctrl+G` for the graph view.
3. Start at `_meta/MOC - Map of Content.md`.

## Exam-prep proofs

Start at `DL-RL-Vault/proofs/INDEX.md`. Each proof is a self-contained note: statement, setup, full derivation, intuition, and "what to remember" annotations on the load-bearing steps.

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

## Roadmap (planned code stages)

- [x] **Stage 1:** stochastic bandits.
- [ ] **Stage 2:** tabular Q-learning on FrozenLake / Taxi.
- [ ] **Stage 3:** DQN on CartPole.
- [ ] **Stage 4:** PPO on LunarLander / Atari.

## Course context

Reference textbooks: Lattimore & Szepesvári, *Bandit Algorithms* (2020); Sutton & Barto, *Reinforcement Learning: An Introduction* (2018); Goodfellow, Bengio, Courville, *Deep Learning* (2016). Topics covered: bandits (stochastic, contextual, adversarial), online learning (FTL/FTRL/MD), zero-sum games, MDPs, value iteration, policy gradients, TRPO/PPO, RLHF/DPO, plus deep-learning fundamentals.
