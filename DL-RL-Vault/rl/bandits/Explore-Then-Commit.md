# Explore-Then-Commit (ETC)

**Tags:** #bandits #algorithm

Simplest sublinear-regret [[Stochastic Bandits|bandit algorithm]].

## Algorithm

Parameter $T_0$.
1. **Explore:** pull each arm $T_0$ times (total $KT_0$ rounds).
2. **Commit:** at round $\tau = KT_0 + 1$, compute $\hat a = \arg\max_a \hat\mu_\tau(a)$.
3. Play $\hat a$ for all remaining rounds.

## Guarantee

With $T_0 = (T/K)^{2/3}(\log T)^{1/3}$:
$$R^T_\mathcal{D} = \widetilde{O}(K^{1/3} T^{2/3}).$$

## Analysis (two arms)

Three regret sources, decomposed in the annotated bound:
$$R^T \leq \underbrace{KT_0}_{\text{EXPL}} + \underbrace{(1 - 4e^{-2\epsilon^2 T_0}) \cdot 2\epsilon(T-KT_0)}_{\text{COMMIT (good case)}} + \underbrace{4e^{-2\epsilon^2 T_0}(T-KT_0)}_{\text{BAD ESTIMATES}}.$$

- **EXPL:** pay $\Delta_a$ each of the $T_0$ pulls of the suboptimal arm.
- **COMMIT (good case):** [[Hoeffding Inequality]] + union bound says both empirical means are within $\epsilon$ of truth with prob $\geq 1 - 4e^{-2\epsilon^2 T_0}$; in that case the gap $\hat\mu_\tau(a_2) - \hat\mu_\tau(a_1) \leq 2\epsilon$, so committing to a sub-optimal arm costs at most $2\epsilon$ per round.
- **BAD ESTIMATES:** the small-probability event where Hoeffding fails — bounded trivially by $T - KT_0$.

Balance $\epsilon = \sqrt{\log T / T_0}$ and pick $T_0$ optimally → $\widetilde{O}(T^{2/3})$.

## Limitations

- The $T^{2/3}$ exponent is **suboptimal** — the [[Bandit Lower Bound]] is $\sqrt{KT}$.
- Requires knowing $T$ in advance to set $T_0$.
- Instance-independent bound; better $\Delta$-aware bounds exist (Lattimore & Szepesvári, Ch. 6).

→ Next algorithm: **UCB** closes the gap.
