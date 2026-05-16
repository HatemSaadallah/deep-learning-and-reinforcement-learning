# Proof — Lipschitz Contextual Bandits

**Tags:** #proof #bandits #contextual
**Topic:** Regret bound for [[Contextual Bandit|contextual bandits]] with Lipschitz reward.

## Statement

Setting: contexts $x \in \mathcal{X} \subset [0,1]^d$, $K$ arms, expected reward $\mu(x, a)$ is $L$-Lipschitz in $x$ for every arm. Rewards in $[0, 1]$.

**Discretization-based algorithm (Uniform discretization + UCB per cell):** partition $\mathcal{X}$ into a uniform $\epsilon$-net of $N = (1/\epsilon)^d$ cells. Within each cell, run an independent UCB1 instance.

**Regret bound:**
$$\boxed{\;R^T \;\leq\; \widetilde{O}\!\left(T^{(d+1)/(d+2)}\right).\;}$$
The exponent interpolates between $\sqrt{T}$ (when $d=0$, recovers regular bandits) and linear (as $d \to \infty$).

## Setup

- $\mathcal{X}$: context space, $d$-dimensional unit cube.
- $\mathcal{C}_1, \dots, \mathcal{C}_N$: cells of an $\epsilon$-uniform grid. $N = \Theta((1/\epsilon)^d)$. Each cell has diameter $\leq \sqrt{d}\,\epsilon = O(\epsilon)$.
- $T_C$: number of rounds with $x_t \in \mathcal{C}$. So $\sum_C T_C = T$.
- For each cell $\mathcal{C}$, treat all contexts in $\mathcal{C}$ as a single "averaged" bandit instance with means $\bar\mu_\mathcal{C}(a) := \mathbb{E}_{x \sim \text{rounds in } \mathcal{C}}[\mu(x, a)]$.

## Proof

**Step 1 — Decompose regret into per-cell regret + Lipschitz error.**

For each round $t$ with $x_t \in \mathcal{C}$:
$$\mu(x_t, a^*(x_t)) - \mu(x_t, a_t) \;=\; \underbrace{\bigl[\bar\mu_\mathcal{C}(\bar a_\mathcal{C}^*) - \bar\mu_\mathcal{C}(a_t)\bigr]}_{\text{within-cell regret}} + \underbrace{\bigl[\mu(x_t, a^*(x_t)) - \bar\mu_\mathcal{C}(\bar a_\mathcal{C}^*)\bigr]}_{\text{Lipschitz error 1}} + \underbrace{\bigl[\bar\mu_\mathcal{C}(a_t) - \mu(x_t, a_t)\bigr]}_{\text{Lipschitz error 2}}.$$

where $\bar a_\mathcal{C}^* := \arg\max_a \bar\mu_\mathcal{C}(a)$ is the best arm averaged over $\mathcal{C}$.

**Lipschitz errors:** since $\mathrm{diam}(\mathcal{C}) \leq \sqrt{d}\,\epsilon$ and $\mu$ is $L$-Lipschitz in $x$:
$$|\mu(x, a) - \bar\mu_\mathcal{C}(a)| \leq L \sqrt{d}\,\epsilon \quad \forall x \in \mathcal{C}.$$
Each Lipschitz error term is $\leq L\sqrt{d}\,\epsilon$, summing per round to at most $2 L \sqrt{d}\,\epsilon T$.

**Step 2 — Within-cell regret via UCB1.**

Within cell $\mathcal{C}$, an instance of UCB1 sees $T_\mathcal{C}$ rounds against a fixed (averaged) bandit. By [[UCB Analysis|UCB instance-independent bound]]:
$$R_\mathcal{C} \;\leq\; O\!\left(\sqrt{K T_\mathcal{C} \log T}\right).$$

(Technicality: the rewards within a cell aren't exactly i.i.d. since contexts vary slightly — but Lipschitz bounds the bias by $L\sqrt{d}\,\epsilon$, already accounted for in Step 1.)

**Step 3 — Sum over cells (Cauchy-Schwarz).**

$$\sum_\mathcal{C} R_\mathcal{C} \;\leq\; \sum_\mathcal{C} c\sqrt{K T_\mathcal{C} \log T} \;\overset{\text{C-S}}{\leq}\; c\sqrt{K \log T}\cdot\sqrt{N}\cdot\sqrt{\sum_\mathcal{C} T_\mathcal{C}} \;=\; c\sqrt{K N T \log T}.$$

With $N = (1/\epsilon)^d$: total within-cell regret $\leq c\,\epsilon^{-d/2}\sqrt{K T \log T}$.

**Step 4 — Combine.**

$$R^T \;\leq\; c\,\epsilon^{-d/2}\sqrt{KT \log T} + 2 L\sqrt{d}\,\epsilon T.$$

**Step 5 — Optimize $\epsilon$.**

Treat $\epsilon$ as a free parameter. Define $A = c\sqrt{KT\log T}$, $B = 2L\sqrt{d}\,T$.
$$f(\epsilon) = A \epsilon^{-d/2} + B\epsilon.$$
$$f'(\epsilon) = -\frac{d}{2} A \epsilon^{-d/2 - 1} + B = 0 \;\Longrightarrow\; \epsilon^{d/2 + 1} = \frac{dA}{2B}.$$

Hence
$$\epsilon^* \;\propto\; \left(\frac{A}{B}\right)^{2/(d+2)} \;=\; \Theta\!\left(T^{-1/(d+2)}\right)\quad \text{(up to logs and constants)}.$$

Plugging back, both terms become $\Theta(T \cdot \epsilon^*) = \Theta(T^{1 - 1/(d+2)}) = \Theta(T^{(d+1)/(d+2)})$:
$$R^T \;\leq\; \widetilde{O}\!\left(T^{(d+1)/(d+2)}\right). \qquad \square$$

## Sanity checks

| $d$ | Exponent | Meaning |
|---|---|---|
| $0$ | $1/2$ | No context → regular bandit, recover $\sqrt{T}$ |
| $1$ | $2/3$ | Same exponent as [[Explore-Then-Commit]] |
| $2$ | $3/4$ | Begins to feel high-dim |
| $d \to \infty$ | $\to 1$ | Curse of dimensionality |

## Intuition / what to remember

- **Two competing errors:** discretization error scales with $\sqrt{N}$ (more cells = more independent bandit problems), Lipschitz approximation error scales with $\epsilon$ (finer cells = smaller bias). The optimal $\epsilon$ balances them.
- **Cauchy-Schwarz is the load-bearing inequality** for combining per-cell regrets, since $\sum_\mathcal{C}\sqrt{T_\mathcal{C}} \leq \sqrt{N \cdot T}$.
- **The exponent $(d+1)/(d+2)$ is sharp** (matching lower bound up to logs). You cannot get $\sqrt{T}$ in $d \geq 1$ without further structure (e.g. linearity → [[LinUCB]] gives $\sqrt{dT}$).
- **Zooming algorithms** (Slivkins) adapt the discretization to the data and improve constants but match the same worst-case rate.

## See also

- [[Contextual Bandit]] — the general framework.
- [[LinUCB]] — assumes linear structure, breaks the curse of dimensionality.
- [[Stochastic Bandits]] — the $d=0$ special case.
- [[UCB Analysis]] — the per-cell building block.
