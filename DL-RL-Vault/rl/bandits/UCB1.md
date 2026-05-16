# UCB1

**Tags:** #bandits #algorithm

The canonical **optimism-in-face-of-uncertainty** algorithm for [[Stochastic Bandits|stochastic bandits]]. Matches the [[Bandit Lower Bound|$\Omega(\sqrt{KT})$ lower bound]] up to log factors — closing the gap left open by [[Explore-Then-Commit|ETC]].

## Algorithm (Auer, Cesa-Bianchi, Fischer 2002)

**Initialization:** play each arm once.

**For** $t = K+1, \dots, T$:
1. Compute the UCB index for every arm:
$$\mathrm{UCB}_t(a) \;=\; \underbrace{\hat\mu_t(a)}_{\text{exploitation}} \;+\; \underbrace{\sqrt{\frac{2 \log T}{N_t(a)}}}_{\text{exploration}}.$$
2. Play $a_t = \arg\max_a \mathrm{UCB}_t(a)$.
3. Observe $r_t$, update $\hat\mu, N$.

## Why optimism

The UCB index is, with high probability, an **upper bound** on the true mean $\mu(a)$ (see [[Clean Event]]). Acting greedily on this upper bound:
- Pulls **high-mean arms** (because their estimated mean is large).
- Pulls **under-explored arms** (because their confidence radius $\sqrt{2 \log T / N_t(a)}$ is large).

Over time, the radius for any pulled arm shrinks like $1/\sqrt{N_t(a)}$, so well-sampled arms stop "winning by uncertainty" and the optimal arm dominates. See the [[Optimism Principle]] for the general recipe.

## Guarantees

| Bound | Rate |
|---|---|
| Instance-dependent | $R^T = O\!\Bigl(\log T \sum_{a:\Delta_a > 0} 1/\Delta_a\Bigr)$ |
| Instance-independent | $R^T = O(\sqrt{KT \log T}) = \widetilde{O}(\sqrt{KT})$ |
| Lower bound | $\Omega(\sqrt{KT})$ — matched up to logs |

Full proof: [[UCB Analysis]].

## Variants

| Variant | Confidence radius | Note |
|---|---|---|
| **UCB1** (this) | $\sqrt{2 \log T / N_t(a)}$ | Needs $T$; instance-indep variant uses $\log t$ for anytime guarantees |
| **UCB-V** (Audibert et al.) | $\sqrt{2 V_t(a) \log T / N_t(a)} + c/N_t(a)$ | Uses empirical variance $V_t(a)$ — tighter for low-variance arms |
| **KL-UCB** (Garivier-Cappé) | KL-based ball: $\{q : N_t(a) \cdot \mathrm{KL}(\hat\mu_t(a), q) \leq \log T\}$ | Tight for Bernoulli arms — matches the constant in the LB |
| **MOSS** (Audibert-Bubeck) | $\sqrt{\log(T/(K N_t(a))) / N_t(a)}$ | Removes a $\log T$ factor in the instance-indep bound |

## Practical notes

- **No hyperparameters** (beyond knowing $T$, which can be replaced by $\log t$).
- **$O(K)$ time per round** to find the argmax.
- **Anytime UCB** uses $\log t$ instead of $\log T$ — slightly looser theory, identical practical behavior, no horizon required.
- **Cold start:** the "play each arm once" prelude means UCB needs at least $K$ rounds to start being interesting; in regimes with $K \sim T$ it pays a heavy upfront cost.

## How it compares

| Algorithm | Knows $\Delta$? | Regret | Implementation |
|---|---|---|---|
| [[Explore-Then-Commit\|ETC]] | No (gap-free variant) | $\widetilde{O}(K^{1/3} T^{2/3})$ | Simple but wrong exponent |
| [[Stochastic Bandits\|$\epsilon$-greedy]] | Often yes (for decaying $\epsilon$) | $O(K\log T / \Delta^2)$ if tuned; $\Theta(T)$ if fixed $\epsilon$ | Simple, brittle |
| **UCB1** | **No** | $\widetilde{O}(\sqrt{KT})$ | Universal — same code works on every instance |
| [[Thompson Sampling]] | No | $\widetilde{O}(\sqrt{KT})$ | Often beats UCB in practice |

The killer feature of UCB: **universality**. The same algorithm matches the optimal ETC parameter on every problem instance, *without knowing the gaps*. See lecture comment slide 14.

## Connection to other vault topics

- [[Clean Event]] — the lemma that powers the analysis.
- [[Optimism Principle]] — the general design recipe.
- [[LinUCB]] — UCB1 with linear function approximation.
- [[Thompson Sampling]] — the Bayesian alternative.
- `code/01_bandits/agents.py::UCB1` — working implementation.

## References

- Auer, Cesa-Bianchi, Fischer, *Finite-time analysis of the multiarmed bandit problem*, Machine Learning 2002.
- Lattimore & Szepesvári, *Bandit Algorithms* (2020), Ch. 7–8.
