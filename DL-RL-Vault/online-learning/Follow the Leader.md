# Follow the Leader (FTL)

**Tags:** #online-learning #algorithm #foundational

## Setting

[[Online Learning]] with rounds $t = 1, \dots, T$:
1. Pick action $x_t \in \mathcal{K}$ (convex set).
2. Adversary reveals loss $f_t$.
3. You incur $f_t(x_t)$.

Objective: minimize [[Regret]]
$$R_T = \sum_{t=1}^T f_t(x_t) - \min_{x \in \mathcal{K}} \sum_{t=1}^T f_t(x).$$

## The rule

At round $t$, play the best action on the past:
$$x_t = \arg\min_{x \in \mathcal{K}} \sum_{s=1}^{t-1} f_s(x).$$

The most natural strategy: "the leader so far is still the leader."

## When FTL works

- **Strongly convex / quadratic losses:** $O(\log T)$ regret. The leader stabilizes quickly.

## When FTL fails

- **Linear losses (online convex setting):** can suffer $\Omega(T)$ regret.
  *Counterexample:* $\mathcal{K} = [-1, 1]$, losses alternate $f_t(x) = \pm x$. Leader flips between $-1$ and $+1$ every round — always exactly wrong. Regret $\sim T$.

- **[[Stochastic Bandits|Bandit setting]] (greedy = FTL):** also $\Omega(T)$.
  *Counterexample (lecture slide 12):* two arms. $r(a_1) \in \{0,1\}$ with prob $1/2$ each; $r(a_2) \equiv 1/4$. With prob $1/2$ the first pull of $a_1$ returns $0$ → $\hat\mu(a_1) = 0 < 1/4 = \hat\mu(a_2)$ → greedy commits to $a_2$ forever. Regret $\Delta_{a_2} = 1/4$ per round → $\Omega(T)$.

The pathology in both cases: FTL is **unstable** near ties — it commits before estimates stabilize. The bandit version is worse: with bandit feedback you never learn about the arm you stopped pulling.

## The fix → [[Follow the Regularized Leader]]

Add a regularizer $R(x)$:
$$x_t = \arg\min_{x \in \mathcal{K}} \left[ \eta \sum_{s=1}^{t-1} f_s(x) + R(x) \right].$$

- $R(x) = \tfrac{1}{2}\|x\|^2$ → [[Online Gradient Descent]]
- $R(x) = \sum_i x_i \log x_i$ → [[Hedge - Multiplicative Weights]]

FTRL achieves $O(\sqrt{T})$ regret for general convex losses.

## Variant → [[Follow the Perturbed Leader]]

Add random noise to past losses instead of a regularizer. Useful for [[Adversarial Bandits]].

## Connection to RL

- [[Adversarial Bandits]] — FTPL is a core building block.
- [[Self-Play and Game-Theoretic RL]] — FTRL dynamics → Nash equilibria in zero-sum games.
- [[PPO]] / [[TRPO]] — the trust region is structurally a KL regularizer → can be read as FTRL on policies.

## References

- Hazan, *Introduction to Online Convex Optimization* — canonical textbook treatment.
- Shalev-Shwartz, *Online Learning and Online Convex Optimization* (2011 survey).
