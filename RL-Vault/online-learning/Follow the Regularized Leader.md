# Follow the Regularized Leader (FTRL)

**Tags:** #online-learning #algorithm

Generalization of [[Follow the Leader]] that fixes its instability by adding a regularizer.

## Rule

$$x_t = \arg\min_{x \in \mathcal{K}} \left[ \eta \sum_{s=1}^{t-1} f_s(x) + R(x) \right]$$

## Regret

For convex losses and well-chosen $R$: $R_T = O(\sqrt{T})$.

## Instances

| Regularizer $R(x)$ | Algorithm |
|---|---|
| $\tfrac{1}{2}\|x\|^2$ | [[Online Gradient Descent]] |
| $\sum_i x_i \log x_i$ (neg-entropy) | [[Hedge - Multiplicative Weights]] |
| KL divergence to prior policy | [[PPO]] / [[TRPO]] (interpretation) |

## See also

- [[Follow the Perturbed Leader]] — noise instead of regularizer.
- [[Mirror Descent]] — dual view of FTRL.
