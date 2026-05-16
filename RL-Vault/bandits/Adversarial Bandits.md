# Adversarial Bandits

**Tags:** #bandits #framework
**Source:** Bocconi RL course (Celli, Lecture 4, 2026-04-01)

[[Stochastic Bandits|Bandits]] where the loss / reward sequence is chosen by an **adversary** (possibly knowing the algorithm), not drawn i.i.d. from a fixed distribution. No statistical assumption on rewards.

## Two feedback models

| Model | What you observe | Lower bound | Optimal algorithm |
|---|---|---|---|
| **Full information** ([[Adversarial Expert Problem]]) | entire loss vector $\ell_t \in [0,1]^K$ | $\Omega(\sqrt{T \log K})$ | [[Hedge - Multiplicative Weights\|Hedge]] |
| **Bandit feedback** (this note) | only $\ell_t(a_t)$ | $\Omega(\sqrt{KT})$ | [[EXP3]] |

The gap between $\sqrt{\log K}$ and $\sqrt{K}$ in the lower bound is **the price of partial information**.

## Setting

At each round $t = 1, \dots, T$:
1. Environment (adversary) chooses loss function $\ell_t: [K] \to [0, 1]$.
2. Learner picks an arm $a_t$ (possibly randomized).
3. Learner suffers $\ell_t(a_t)$ and observes **only** $\ell_t(a_t)$.

[[Regret]]:
$$R_T = \mathbb{E}\!\left[\sum_t \ell_t(a_t)\right] - \min_{a \in [K]} \sum_t \ell_t(a).$$

The minimum is over fixed actions in hindsight (static regret). Stronger notions (dynamic regret, switching regret) exist but are not the focus here.

## Two facts that shape the algorithmic landscape

### Deterministic algorithms suffer linear regret

If the algorithm is deterministic, an oblivious adversary computes the action $a_t$ in advance and **always assigns it loss 1, other arms loss 0**. Then $\sum_t \ell_t(a_t) = T$ while $\min_a \sum_t \ell_t(a) \leq T/K$, so $R_T = \Omega(T)$.

→ **Randomization is mandatory** in the adversarial setting. (Compare: [[UCB1]] is deterministic given the history but works for [[Stochastic Bandits|stochastic]] bandits because the *environment* is stochastic.)

### Cannot apply Hedge directly

[[Hedge - Multiplicative Weights|Hedge]] needs the full loss vector $\ell_t \in [0, 1]^K$ to update weights. In the bandit setting we observe only $\ell_t(a_t)$.

**Fix:** the **importance-weighted estimator**.

## Importance-weighted loss estimator

Define
$$\tilde\ell_t(a) := \begin{cases} \dfrac{\ell_t(a_t)}{x_t(a_t)} & \text{if } a = a_t \\ 0 & \text{otherwise} \end{cases}$$
where $x_t(a) = \Pr[a_t = a]$ under the algorithm's randomization.

**Key property — unbiasedness:**
$$\mathbb{E}_{a_t \sim x_t}[\tilde\ell_t(a)] = x_t(a) \cdot \frac{\ell_t(a)}{x_t(a)} + (1 - x_t(a)) \cdot 0 = \ell_t(a).$$

So $\tilde\ell_t$ is an **unbiased estimator** of the true loss vector — even though we only see one coordinate per round.

**Cost:** the variance is large. $\tilde\ell_t(a) \in \{0, \ell_t(a_t)/x_t(a_t)\}$, which can be up to $1/x_t(a_t)$ — unbounded as $x_t(a_t) \to 0$. This variance is the source of the extra $\sqrt{K}$ factor in the bandit regret bound.

## Algorithm: [[EXP3]]

Plug the importance-weighted estimator into Hedge:
1. Play $a_t \sim x_t$ where $x_t \propto w_t$.
2. Observe $\ell_t(a_t)$.
3. Compute $\tilde\ell_t$ (zero except for arm $a_t$).
4. Multiplicative update: $w_{t+1}(a) = w_t(a) \cdot e^{-\eta \tilde\ell_t(a)}$.

**Regret:** $R_T = O(\sqrt{K T \log K})$ with $\eta = \sqrt{\log K / (KT)}$. Matches the lower bound up to logs.

Full algorithm + analysis: [[EXP3]].

## Why the rate is worse than stochastic bandits

| Setting | Optimal regret |
|---|---|
| [[Stochastic Bandits]] (with gaps) | $O(\log T \sum 1/\Delta_a)$ |
| [[Stochastic Bandits]] (gap-free / worst-case) | $\widetilde{O}(\sqrt{KT})$ |
| **Adversarial bandits** | $\widetilde{O}(\sqrt{KT})$ |

The worst-case rates match — the adversarial setting **cannot be much worse** than the worst stochastic instance. But there is **no instance-dependent improvement** in the adversarial case: gaps $\Delta_a$ aren't well-defined when losses change every round.

## Variants

- **EXP3.P** — high-probability bound instead of in expectation.
- **EXP3-IX** — uses implicit exploration via biasing $\tilde\ell$ to be one-sided; same regret, simpler high-probability analysis.
- **Tsallis-INF** — best-of-both-worlds: $\sqrt{KT}$ adversarial, $\log T$ stochastic, same algorithm.

## See also

- [[EXP3]] — the algorithm note.
- [[Adversarial Expert Problem]] — the full-info sibling.
- [[Hedge - Multiplicative Weights]] — the building block.
- [[Follow the Regularized Leader]] — the framework EXP3 lives in.
- [[Stochastic Bandits]] — the easier sibling.
