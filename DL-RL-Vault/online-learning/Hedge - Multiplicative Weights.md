# Hedge / Multiplicative Weights

**Tags:** #online-learning #algorithm

[[Follow the Regularized Leader]] on the probability simplex with negative-entropy regularizer.

## Rule

Maintain weights $w_t(i)$ over $N$ experts. Update:
$$w_{t+1}(i) = w_t(i) \cdot e^{-\eta f_t(i)}, \qquad p_t(i) = \frac{w_t(i)}{\sum_j w_t(j)}.$$

## Regret

$R_T = O(\sqrt{T \log N})$ — only logarithmic in the number of experts.

## Why it matters

- **Boosting:** AdaBoost is essentially Hedge applied to training examples.
- **Game theory:** Hedge vs. Hedge in zero-sum games converges to Nash → foundation of [[Self-Play and Game-Theoretic RL]].
- **RL:** core primitive for [[Adversarial Bandits]] (Exp3).
