# $\epsilon$-Greedy

**Tags:** #bandits #algorithm

The simplest exploration heuristic: most of the time exploit, occasionally explore at random. Pedagogically central but **provably linear-regret** in its naive form.

## Algorithm (Sutton 1988)

Parameter $\epsilon \in [0, 1]$.

**For** $t = 1, 2, \dots$:
- With probability $\epsilon$: $a_t \sim \mathcal{U}([K])$ (uniform random exploration).
- With probability $1 - \epsilon$: $a_t = \arg\max_a \hat\mu_t(a)$ (greedy / exploit).

Update empirical means after observing reward.

## Linear regret (fixed $\epsilon$)

For any fixed $\epsilon > 0$ and any [[Stochastic Bandits|stochastic bandit]] instance:
$$\boxed{\;R^T \;\geq\; \epsilon \cdot \frac{K-1}{K} \cdot \Delta_{\min}\, T.\;}$$

**Why:** during the $\epsilon$-fraction of rounds where the algorithm explores uniformly, each suboptimal arm has equal $1/K$ chance of being pulled. The contribution to expected per-round regret from exploration is at least
$$\epsilon \cdot \frac{1}{K}\sum_{a: \Delta_a > 0} \Delta_a \;\geq\; \epsilon \cdot \frac{K-1}{K}\,\Delta_{\min}.$$
This regret accumulates linearly. **No amount of exploitation can fix this** — pure-greedy phases are zero-regret only on the optimal arm, but explore phases bleed regret deterministically.

## Decaying $\epsilon$ (slightly better)

Set $\epsilon_t = \min\!\bigl(1,\; K / (d^2 t)\bigr)$ for some constant $d \leq \Delta_{\min}$. Then
$$R^T = O\!\left(\frac{K \log T}{d^2}\right).$$

This is **logarithmic in $T$** — comparable to [[UCB1]] up to constants — but with a fatal caveat:

> **Requires knowledge of (a lower bound on) $\Delta_{\min}$** — exactly the unknown quantity we're trying to learn.

So gap-free $\epsilon$-greedy has two failure modes:
- Fixed $\epsilon$: provably linear regret.
- Decaying $\epsilon$: requires $\Delta_{\min}$ to tune properly.

## Why it's a great pedagogical baseline

- **Simplest possible exploration strategy.** Two lines of code.
- **Demonstrates the necessity of "smart" exploration**: $\epsilon$-greedy explores stupidly (uniform random), which is why it can't compete with UCB / Thompson on regret.
- **Often used in deep RL** — DQN's exploration policy is decaying-$\epsilon$ greedy, with $\epsilon$ annealed over millions of steps from $\sim 1$ to $\sim 0.01$. The fact that this works (despite poor theory) is more about *neural-net implicit exploration* than the $\epsilon$-greedy schedule.

## Comparison

| Algorithm | Regret rate | Tuning |
|---|---|---|
| **$\epsilon$-greedy** (fixed) | $\Theta(T)$ | $\epsilon$ |
| **$\epsilon$-greedy** (decaying) | $O(K \log T / \Delta^2)$ | $\epsilon_t = \min(1, K/(d^2 t))$, needs $d \leq \Delta_{\min}$ |
| [[Explore-Then-Commit\|ETC]] (gap-free) | $\widetilde{O}(K^{1/3} T^{2/3})$ | $T_0$, knows only $T, K$ |
| [[UCB1]] | $\widetilde{O}(\sqrt{KT})$ | nothing |
| [[Thompson Sampling]] | $\widetilde{O}(\sqrt{KT})$ | prior |

## Where it shows up in this vault

- `code/01_bandits/agents.py::EpsilonGreedy` — implemented.
- The empirical regret plot in `code/01_bandits/regret_curves.png` shows it sublinear *on this instance* (with fixed $\epsilon = 0.1$ over $T = 5000$) — but **only because the instance is not adversarial**. On a harder instance with tiny $\Delta$, the linear-regret pathology would be visible.

## See also

- [[Stochastic Bandits]] — the setting.
- [[UCB1]] / [[Thompson Sampling]] — the principled alternatives.
- [[Optimism Principle]] — the meta-recipe that $\epsilon$-greedy *does not* follow.
