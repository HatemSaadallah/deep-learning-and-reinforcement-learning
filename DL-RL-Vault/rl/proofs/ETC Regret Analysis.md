# Proof — ETC Regret $\widetilde{O}(K^{1/3} T^{2/3})$

**Tags:** #proof #bandits #etc
**Topic:** Regret bound for [[Explore-Then-Commit]].

## Statement

For $K$-armed [[Stochastic Bandits|stochastic bandits]] with rewards in $[0,1]$, ETC with $T_0 = (T/K)^{2/3}(\log T)^{1/3}$ satisfies
$$\boxed{\;R^T_\mathcal{D} \;\leq\; \widetilde{O}\bigl(K^{1/3} T^{2/3}\bigr).\;}$$

## Setup

- Algorithm: pull each arm $T_0$ times (so $KT_0$ exploration rounds), then commit to $\hat a = \arg\max_a \hat\mu_\tau(a)$ for $\tau = KT_0$.
- Two-arm analysis (WLOG $a_1$ optimal, gap $\Delta = \Delta_{a_2}$). The $K$-arm case is similar — just union-bound over arms.
- Tool: [[Hoeffding Inequality]] — $\Pr[|\hat\mu_n - \mu| \geq \epsilon] \leq 2 e^{-2\epsilon^2 n}$.

## Proof

**Step 1 — Decompose regret by phase.**

$$R^T \;=\; \underbrace{R^{\text{expl}}}_{\text{exploration}} + \underbrace{R^{\text{commit}}}_{\text{commitment}}.$$

Exploration regret: each arm is pulled $T_0$ times.
$$R^{\text{expl}} = T_0 \sum_a \Delta_a \leq 2 T_0 \quad (\text{two arms, each } \Delta_a \leq 1).$$
For $K$ arms generally: $R^{\text{expl}} \leq K T_0$.

**Step 2 — Concentration via Hoeffding + union bound.**

Apply Hoeffding to each arm's $T_0$ samples:
- $\Pr[\hat\mu_\tau(a_1) < \mu_1 - \epsilon] \leq e^{-2\epsilon^2 T_0}$.
- $\Pr[\hat\mu_\tau(a_2) > \mu_2 + \epsilon] \leq e^{-2\epsilon^2 T_0}$.
- And the symmetric tails.

**Union bound:** define the **good event** $G_\epsilon = \{\hat\mu_\tau(a_1) \geq \mu_1 - \epsilon \text{ and } \hat\mu_\tau(a_2) \leq \mu_2 + \epsilon\}$. Then
$$\Pr[\neg G_\epsilon] \leq 4 e^{-2\epsilon^2 T_0}.$$

**Step 3 — Commit phase under $G_\epsilon$ (good case).**

If we end up committing to the sub-optimal arm $a_2$, then $\hat\mu_\tau(a_2) \geq \hat\mu_\tau(a_1)$. Under $G_\epsilon$:
$$\mu_2 + \epsilon \;\geq\; \hat\mu_\tau(a_2) \;\geq\; \hat\mu_\tau(a_1) \;\geq\; \mu_1 - \epsilon.$$
So $\mu_1 - \mu_2 \leq 2\epsilon$, i.e. $\Delta \leq 2\epsilon$.

**Conclusion:** under $G_\epsilon$, committing to a suboptimal arm costs at most $2\epsilon$ per commit-round. Commit regret (good case): at most $2\epsilon(T - KT_0)$.

**Step 4 — Commit phase under $\neg G_\epsilon$ (bad case).**

We may have committed to an arm of arbitrary suboptimality. Worst-case regret per round is $1$, so total bad-case commit regret $\leq (T - KT_0)$.

**Step 5 — Total bound.**

$$R^T \leq \underbrace{KT_0}_{\text{expl}} + \underbrace{2\epsilon (T - KT_0)}_{\text{good commit}} + \underbrace{4e^{-2\epsilon^2 T_0}(T - KT_0)}_{\text{bad commit}}.$$

**Step 6 — Choose $\epsilon$ to kill the bad-case term.**

Take $\epsilon = \sqrt{\log T / T_0}$. Then $e^{-2\epsilon^2 T_0} = e^{-2\log T} = T^{-2}$, so the third term is $\leq 4/T$. Negligible.

$$R^T \;\leq\; K T_0 \;+\; 2T\sqrt{\frac{\log T}{T_0}} \;+\; O(1/T).$$

**Step 7 — Optimize $T_0$.**

Differentiate the first two terms w.r.t. $T_0$:
$$K - T\sqrt{\log T} \cdot T_0^{-3/2} = 0 \;\Longrightarrow\; T_0^{3/2} = \frac{T\sqrt{\log T}}{K} \;\Longrightarrow\; T_0 = \left(\frac{T}{K}\right)^{2/3} (\log T)^{1/3}.$$

Plug back: both terms become $K^{1/3} T^{2/3} (\log T)^{1/3}$:
$$R^T \leq 3 K^{1/3} T^{2/3} (\log T)^{1/3} = \widetilde{O}\bigl(K^{1/3} T^{2/3}\bigr). \qquad \square$$

## Intuition / what to remember

- **Three regret sources** (the lecture annotates these as EXPL / COMMIT / BAD ESTIMATES on slide 17): exploration cost, residual commit cost when estimates are good, and a small probability of bad estimates.
- **The $T^{2/3}$ exponent comes from balancing $T_0$ vs $T/\sqrt{T_0}$.** Equating $T_0 \sim T/\sqrt{T_0}$ gives $T_0 \sim T^{2/3}$.
- **This exponent is suboptimal** — the [[Bandit Lower Bound|lower bound]] is $\sqrt{KT}$. UCB achieves $\widetilde{O}(\sqrt{KT})$ by not committing.
- **Instance-dependent variant:** if $\Delta$ is known, take $T_0 = (4/\Delta^2) \log T$ → $R^T = O(\log T / \Delta)$. The horizon-only bound above is *gap-free*.

## Key tools used

| Tool | Where |
|---|---|
| [[Hoeffding Inequality]] | Steps 2 |
| Union bound | Step 2 |
| Optimization of free parameters | Steps 6–7 |
