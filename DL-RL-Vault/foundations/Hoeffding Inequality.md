# Hoeffding's Inequality

**Tags:** #concentration #foundational #probability

The workhorse concentration bound for bounded i.i.d. random variables.

## Statement (bounded version)

Let $r_1, \dots, r_n \in [0,1]$ be i.i.d. with mean $\mu$. For any $\epsilon > 0$:
$$\Pr\left[\left|\frac{1}{n}\sum_{i=1}^n r_i - \mu\right| \geq \epsilon\right] \leq 2e^{-2\epsilon^2 n}.$$

The probability of being far from the mean decreases **exponentially in $\epsilon^2 n$**.

## Inverted form (the way you actually use it)

With probability $\geq 1 - \delta$:
$$|\hat\mu - \mu| \leq \sqrt{\frac{\log(2/\delta)}{2n}}.$$

This $\sqrt{\log(1/\delta)/n}$ "confidence radius" is what shows up in UCB.

## Where it appears in bandits

- [[Explore-Then-Commit]] analysis: bounds the prob of picking the wrong arm.
- **UCB**: defines the upper confidence bound exactly via this inverted form.
- Any "high-probability" regret bound.

## See also

- [[KL Divergence]] — gives tighter bounds for Bernoulli arms (KL-UCB).
- [[Pinskers Inequality]] — relates KL to total variation, used for [[Bandit Lower Bound|lower bounds]].
