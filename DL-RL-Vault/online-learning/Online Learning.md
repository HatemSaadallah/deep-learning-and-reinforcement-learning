# Online Learning

**Tags:** #foundational #framework

Sequential decision-making under uncertainty, evaluated by [[Regret]] rather than i.i.d. statistical risk.

## Protocol

For $t = 1, \dots, T$:
1. Learner picks $x_t \in \mathcal{K}$.
2. Adversary (or environment) reveals loss $f_t$.
3. Learner incurs $f_t(x_t)$ and observes feedback.

## Feedback models

- **Full information:** observe entire $f_t$ → [[Follow the Leader]], [[Follow the Regularized Leader]].
- **Bandit:** observe only $f_t(x_t)$ → [[Adversarial Bandits]], [[Follow the Perturbed Leader]].

## Connection to RL

RL is a strict generalization: state transitions add temporal credit assignment on top of online learning's per-round structure. Many RL algorithms (policy gradients with trust regions, exploration bonuses) inherit ideas directly from this literature.
