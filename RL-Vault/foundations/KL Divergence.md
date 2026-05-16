# KL Divergence

**Tags:** #information-theory #foundational #definition

## Definition

For distributions $p, q$ on finite $\Omega$:
$$\mathrm{KL}(p, q) := \sum_{\omega \in \Omega} p(\omega) \ln \frac{p(\omega)}{q(\omega)} = \mathbb{E}_p\!\left[\ln \frac{p(\omega)}{q(\omega)}\right].$$

## Key properties

1. **Non-negativity:** $\mathrm{KL}(p, q) \geq 0$, equality iff $p = q$.
2. **Asymmetric:** $\mathrm{KL}(p, q) \neq \mathrm{KL}(q, p)$ in general — not a metric.
3. **Chain rule for products:** if $p = p_1 \times \cdots \times p_n$ and $q = q_1 \times \cdots \times q_n$:
$$\mathrm{KL}(p, q) = \sum_{i=1}^n \mathrm{KL}(p_i, q_i).$$
4. **[[Pinskers Inequality]]:** $2(p(A) - q(A))^2 \leq \mathrm{KL}(p, q)$ for any event $A$.

## KL between biased coins

Let $\mathrm{RC}_\epsilon = \text{Bernoulli}((1+\epsilon)/2)$. For $\epsilon \in (0, 1/2)$:
- $\mathrm{KL}(\mathrm{RC}_\epsilon, \mathrm{RC}_0) \leq 2\epsilon^2$
- $\mathrm{KL}(\mathrm{RC}_0, \mathrm{RC}_\epsilon) \leq \epsilon^2$

These two facts power the [[Bandit Lower Bound]] proof.

## Where it appears in RL

- [[Bandit Lower Bound]] — measure-change argument.
- KL-UCB — tighter confidence intervals for Bernoulli arms.
- [[PPO]] / [[TRPO]] — KL divergence as trust-region constraint between policies.
- Soft Actor-Critic, entropy regularization — KL between policy and prior.
