# EXP3

**Tags:** #bandits #algorithm
**Source:** Auer, Cesa-Bianchi, Freund, Schapire 2002. Celli Lecture 4, slide 19.

The canonical algorithm for [[Adversarial Bandits]]: combines [[Hedge - Multiplicative Weights|Hedge]] with the **importance-weighted loss estimator** to handle bandit feedback.

## Algorithm

**Input:** arms $A = [K]$, horizon $T$, learning rate $\eta$.

**Init:** $w_1(a) = 1$ for all $a$.

**For** $t = 1, \dots, T$:
1. $x_t(a) = \dfrac{w_t(a)}{\sum_{a'} w_t(a')}$ for every $a$.
2. Sample $a_t \sim x_t$.
3. Play $a_t$, observe $\ell_t(a_t)$.
4. Construct importance-weighted estimator:
$$\tilde\ell_t(a) = \frac{\ell_t(a_t)}{x_t(a_t)}\, \mathbb{1}[a_t = a].$$
5. Multiplicative update: $w_{t+1}(a) = w_t(a) \cdot e^{-\eta \tilde\ell_t(a)}$.

## Regret guarantee

With $\eta = \sqrt{\log K / (K T)}$:
$$\boxed{\;R_T \leq O\bigl(\sqrt{K T \log K}\bigr).\;}$$

Matches the $\Omega(\sqrt{KT})$ [[Adversarial Bandits|lower bound]] up to a $\sqrt{\log K}$ factor.

## Why it works (proof outline)

EXP3 is **Hedge run on the estimated loss vectors $\tilde\ell_t$**. Hedge's regret guarantee applies directly:
$$\mathbb{E}\!\left[\sum_t \langle \tilde\ell_t, x_t\rangle - \min_a \sum_t \tilde\ell_t(a)\right] \leq O(\sqrt{T \log K})\quad\text{(in some norm)}.$$

To convert this to a bound on the actual regret, use unbiasedness $\mathbb{E}[\tilde\ell_t(a)] = \ell_t(a)$:
- LHS in expectation equals the true expected regret.
- The variance of $\tilde\ell_t$ enters the bound via $\sum_t \mathbb{E}[\|\tilde\ell_t\|_*^2]$. Computing:
$$\mathbb{E}\!\left[\sum_a \tilde\ell_t(a)^2\right] = \sum_a x_t(a) \cdot \frac{\ell_t(a)^2}{x_t(a)^2} \leq \sum_a \frac{1}{x_t(a)} \cdot 1.$$
But also $\mathbb{E}[\tilde\ell_t(a)^2 | a_t = a] = \ell_t(a)^2 / x_t(a)$, and summed: $\sum_a \ell_t(a)^2 \cdot x_t(a) / x_t(a)^2 \cdot x_t(a) = \sum_a \ell_t(a) \leq K$.

This gives the extra $K$ factor inside the square root, hence $\sqrt{KT \log K}$ instead of Hedge's $\sqrt{T \log K}$.

## Key insights

- **Randomization is mandatory** — see [[Adversarial Bandits]] for why deterministic fails.
- **Importance weighting is the only way to get an unbiased estimator** from bandit feedback. The cost is variance.
- The extra $\sqrt{K}$ over [[Hedge - Multiplicative Weights|Hedge]] is **the price of partial information**.
- **No instance-dependent gains:** no concept of $\Delta_a$ in the adversarial setting, so no log-regret regime.

## Compared to UCB1

| | [[UCB1]] | EXP3 |
|---|---|---|
| Setting | [[Stochastic Bandits\|stochastic]] | [[Adversarial Bandits\|adversarial]] |
| Selection | deterministic argmax | random sample from distribution |
| Estimator | empirical mean | importance-weighted |
| Regret | $\widetilde{O}(\sqrt{KT})$ | $\widetilde{O}(\sqrt{KT \log K})$ |

UCB1 is broken by adversarial losses; EXP3 still works on stochastic but with worse constants.

## Variants

- **EXP3.P** — high-probability bound (instead of expected regret).
- **EXP3-IX** — implicit explicit exploration; biases $\tilde\ell$ to remove the variance blow-up.
- **EXP4** — for contextual / experts setting with $N$ experts: regret $O(\sqrt{KT \log N})$.

## See also

- [[Adversarial Bandits]] — the setting hub.
- [[Hedge - Multiplicative Weights]] — the building block.
- [[Follow the Regularized Leader]] — EXP3 is FTRL with negative-entropy regularizer on the simplex applied to estimated losses.
