# Martingales and Bernstein's Inequality

**Tags:** #foundational #concentration

Two concentration tools that go **beyond [[Hoeffding Inequality|Hoeffding]]**:
- **Bernstein's inequality** — exploits *variance* for sharper bounds when r.v.s have small variance relative to their range.
- **Martingales + Azuma-Hoeffding** — drop the i.i.d. requirement, handle dependent sequences via a "martingale difference" decomposition.

Both are essential for tight analysis of online RL algorithms (UCB-VI, Q-learning-UCB), where the data is adaptively gathered.

## Bernstein's inequality

Let $X_1, \dots, X_n$ be independent with $|X_i - \mathbb{E}[X_i]| \leq R$ and $\mathrm{Var}(X_i) \leq \sigma_i^2$. For any $\epsilon \geq 0$:
$$\Pr\!\left[\sum_{i=1}^n (X_i - \mathbb{E}[X_i]) \geq \epsilon\right] \;\leq\; \exp\!\left\{\frac{-\epsilon^2}{2\sum_i \sigma_i^2 + \tfrac{2}{3} R \epsilon}\right\}.$$

**Inverted form (i.i.d., $\mathrm{Var}(X) \leq \sigma^2$):** w.p. $\geq 1 - \delta$,
$$\frac{1}{n}\sum_i X_i - \mathbb{E}[X] \;\leq\; \widetilde{O}\!\left(\frac{\sigma}{\sqrt{n}} + \frac{R}{n}\right).$$

**Comparison to Hoeffding:** Hoeffding gives $\widetilde{O}(R/\sqrt{n})$ — the full range $R$. When $\sigma \ll R$, Bernstein's $\sigma/\sqrt{n}$ is far sharper. **For large $n$, $\sigma$ dominates.**

## Where Bernstein matters in RL

- **VI / Q-learning sample complexity**: Bernstein-based analysis of value iteration gives optimal $\widetilde{O}(H^4 SA / \epsilon^2)$ — the Hoeffding version gives $H^5$. The win is because **$V$-functions have small variance** (bounded by $V_{\max}$ even if values can range up to $H$).
- **UCB-V** bandit algorithm — uses empirical-variance bonus.

## Martingales

A sequence $\{X_k\}$ adapted to a filtration $\{\mathcal{F}_k\}$ is a **martingale** if:
$$\mathbb{E}[X_k \mid \mathcal{F}_{k-1}] = X_{k-1} \quad \forall k \geq 1.$$

"Conditional on the past, the expected value doesn't drift." The classic example: random walk $S_n = \sum_{i \leq n} X_i$ with $X_i \in \{\pm 1\}$ independent and $\mathbb{E}[X_i] = 0$.

**Martingale difference sequence (MDS):** $D_k := X_k - X_{k-1}$. Satisfies $\mathbb{E}[D_k \mid \mathcal{F}_{k-1}] = 0$.

## Why martingales matter for RL

Consider any function $f: \mathbb{R}^n \to \mathbb{R}$ of dependent variables (e.g. cumulative regret over an online trajectory). Decompose:
$$f(X) - \mathbb{E} f(X) = \sum_{k=1}^n \underbrace{\bigl(\mathbb{E}[f(X) \mid \mathcal{F}_k] - \mathbb{E}[f(X) \mid \mathcal{F}_{k-1}]\bigr)}_{D_k}.$$

Each $D_k$ is a martingale difference. If they're bounded, **Azuma-Hoeffding** gives concentration of $f(X)$ around its mean — *without i.i.d. assumptions*.

This is essential in online RL because actions depend on history, so transitions are not i.i.d. — but martingale differences (e.g. $V^*(s') - \mathbb{E}[V^*(s')]$) are.

## Azuma-Hoeffding inequality

Let $\{D_i\}_{i=1}^n$ be a martingale difference sequence, with $D_i$ supported on an interval of length $R_i$. Then:
$$\Pr\!\left[\sum_{i=1}^n D_i \geq t\right] \;\leq\; \exp\!\left\{-\frac{2t^2}{\sum_{i=1}^n R_i^2}\right\}.$$

This is **structurally identical to Hoeffding**, but applies to martingale differences instead of i.i.d. variables.

## A worked example

Let $\{X_i\} \subset \{\pm 1\}$ be i.i.d. with mean 0. Define $S_n = \sum X_i$. Then $S_n$ is a martingale:
$$\mathbb{E}[S_n \mid S_0, \dots, S_{n-1}] = \mathbb{E}[S_{n-1} + X_n \mid S_{n-1}] = S_{n-1} + \mathbb{E}[X_n] = S_{n-1}.$$

By Azuma-Hoeffding with $R_i = 2$: $\Pr[S_n \geq t] \leq \exp(-t^2 / (2n))$, recovering the classical i.i.d. Hoeffding.

## See also

- [[Hoeffding Inequality]] — the i.i.d. baseline.
- [[VI Generative Setting]] — analysis uses Hoeffding (Bernstein would tighten).
- [[Exploration in RL]] — online algorithms whose analysis crucially uses Azuma-Hoeffding.
- [[Q-learning]] — generative analysis is Hoeffding-based; online version uses Bernstein/Azuma.
