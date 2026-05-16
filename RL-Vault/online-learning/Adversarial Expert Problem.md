# Adversarial Expert Problem

**Tags:** #online-learning #framework

The **full-information** adversarial sibling of [[Adversarial Bandits]]. Choose a distribution over $K$ experts, observe the full loss vector after the choice.

## Setting

At each round $t = 1, \dots, T$:
1. Environment chooses loss vector $\ell_t \in [0, 1]^K$.
2. Learner chooses distribution $x_t \in \Delta_K$ over arms.
3. Learner observes the **full** loss vector $\ell_t$ and incurs $\langle \ell_t, x_t\rangle$.

[[Regret]]:
$$R_T = \sum_t \langle \ell_t, x_t\rangle - \min_{x \in \Delta_K} \sum_t \langle \ell_t, x\rangle = \sum_t \langle \ell_t, x_t\rangle - \min_{a \in [K]} \sum_t \ell_t(a)$$
(the min over $\Delta_K$ is attained at a vertex — best fixed expert in hindsight).

## Lower bound

$$R_T \geq \Omega\!\left(\sqrt{T \log K}\right).$$

The $\log K$ — not $K$ — is the payoff of seeing the full loss vector each round.

## Optimal algorithm: [[Hedge - Multiplicative Weights|Hedge]]

Maintain weights $w_t(a) \in \mathbb{R}_{\geq 0}$. Play $x_t \propto w_t$. After seeing $\ell_t$:
$$w_{t+1}(a) = w_t(a) \cdot e^{-\eta \ell_t(a)}.$$

With $\eta = \sqrt{\log K / T}$: $R_T \leq O(\sqrt{T \log K})$. Matches the lower bound.

## Why deterministic algorithms fail

[[Follow the Leader]] is deterministic: $a_t = \arg\min_a \sum_{\tau < t} \ell_\tau(a)$. Adversary knows this and sets:
- Round 1: $\ell_1 = (0, 0)$ — FTL picks arm 1 by tie-break.
- Round 2: $\ell_2 = (1, 0)$ — FTL still picks arm 1.
- ... adversary can keep loading loss on the chosen arm, making FTL incur $\Omega(T)$ regret.

→ **Randomization is essential** (same conclusion as in [[Adversarial Bandits]]).

## Why OGD on the simplex is suboptimal here

[[OGD Regret Bound|Online gradient descent]] applied to this problem: convex set $\mathcal{C} = \Delta_K$ (diameter $D = O(1)$ in $\ell_2$), Lipschitz constant $L = \|\ell_t\|_2 \leq \sqrt{K}$:
$$R_T = O(LD\sqrt{T}) = O(\sqrt{KT}).$$

This has **the wrong geometry**: the $\sqrt{K}$ instead of $\sqrt{\log K}$ is because $\ell_2$ doesn't respect the simplex structure. The fix is [[Mirror Descent Analysis|mirror descent]] with negative-entropy regularizer — which **recovers Hedge** and gives the optimal $\sqrt{T \log K}$.

## Position in the vault

| Setting | Feedback | Lower bound | Algorithm |
|---|---|---|---|
| **Adversarial Expert Problem** (this) | full $\ell_t$ | $\Omega(\sqrt{T \log K})$ | [[Hedge - Multiplicative Weights\|Hedge]] |
| [[Adversarial Bandits]] | only $\ell_t(a_t)$ | $\Omega(\sqrt{KT})$ | [[EXP3]] |
| [[Online Convex Optimization]] (general) | full $\ell_t$ convex | $\Omega(\sqrt{T})$ | [[OGD Regret Bound\|OGD]], [[Follow the Regularized Leader\|FTRL]], [[Mirror Descent Analysis\|MD]] |

The expert problem is the linear-loss instance of OCO on the simplex.

## See also

- [[Hedge - Multiplicative Weights]] — the canonical algorithm.
- [[Adversarial Bandits]] / [[EXP3]] — bandit-feedback variant.
- [[Mirror Descent Analysis]] — explains why entropy regularization gives optimal rates here.
