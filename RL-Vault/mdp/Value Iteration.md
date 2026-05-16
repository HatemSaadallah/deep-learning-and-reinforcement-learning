# Value Iteration (VI)

**Tags:** #mdp #algorithm #planning

Direct DP algorithm that computes $V^*$ (and the optimal policy $\pi^*$) by iterating the [[Bellman Equations|Bellman optimality equation]] backward in time.

**Setting:** planning — $\mathbb{P}, r$ are known. (For unknown-$\mathbb{P}$ versions, see [[VI Generative Setting]] and online RL algorithms.)

## Algorithm (finite-horizon)

**Input:** MDP $(\mathcal{S}, \mathcal{A}, r, \mathbb{P}, T)$.

**Initialize:** $V_{T+1}^*(s) = 0$ for all $s$.

**For** $t = T, T-1, \dots, 1$:
1. $Q_t^*(s, a) \leftarrow r_t(s, a) + (\mathbb{P}_t V_{t+1}^*)(s, a)$ for all $(s, a)$.
2. $V_t^*(s) \leftarrow \max_a Q_t^*(s, a)$ for all $s$.

**Output:** $\pi_t^*(s) = \arg\max_a Q_t^*(s, a)$ for all $(s, t)$.

## Complexity

$O(T \cdot |\mathcal{S}|^2 \cdot |\mathcal{A}|)$ — linear in horizon, quadratic in states (from the expectation over $s'$), linear in actions. Compare to naive trajectory enumeration: $\Theta((|\mathcal{S}||\mathcal{A}|)^T)$. The exponential-to-polynomial reduction comes entirely from Bellman.

## Infinite-horizon discounted version

Iterate the [[Bellman Equations|Bellman optimality operator]] $\mathcal{T}$ until convergence:
$$V^{(k+1)} \leftarrow \mathcal{T} V^{(k)}, \qquad \mathcal{T} V(s) = \max_a\Bigl[r(s,a) + \gamma \mathbb{E}_{s' \sim \mathbb{P}(\cdot|s,a)} V(s')\Bigr].$$

Since $\mathcal{T}$ is a $\gamma$-contraction in $\|\cdot\|_\infty$ (Banach fixed-point theorem):
$$\|V^{(k)} - V^*\|_\infty \leq \gamma^k \|V^{(0)} - V^*\|_\infty.$$

To achieve $\|V^{(k)} - V^*\|_\infty \leq \epsilon$, need $k = O\!\bigl(\frac{1}{1-\gamma}\log \frac{V_{\max}}{\epsilon}\bigr)$ iterations — exactly the **effective horizon** of the discounted setting.

## When $\mathbb{P}$ is unknown

Replace expectations with sample averages — this is the **generative-model VI** analyzed in [[VI Generative Setting]]. Sample complexity: $\widetilde{O}\bigl(\frac{1}{(1-\gamma)^4 \epsilon^2}\bigr)$ per $(s,a)$ for $\epsilon$-optimal value (Hoeffding-based) or $\widetilde{O}\bigl(\frac{1}{(1-\gamma)^3 \epsilon^2}\bigr)$ with Bernstein.

For **online** RL (only one trajectory at a time, no resets), VI is generalized to UCB-VI / UCRL-style algorithms that combine VI with exploration bonuses.

## VI vs. Policy Iteration

| | Per iteration cost | Iterations to converge |
|---|---|---|
| **VI** | $O(SA)$ Bellman backups | $O\bigl(\tfrac{1}{1-\gamma} \log \tfrac{1}{\epsilon}\bigr)$ |
| **[[Policy Iteration]]** | $O(S^3)$ to solve linear system + $O(SA)$ greedy update | Far fewer iterations (often finite) |

VI is **simple and parallelizable**; PI converges in fewer iterations but each iteration is more expensive. In practice modified PI (truncated policy evaluation + greedy improvement) interpolates between them.

## See also

- [[Markov Decision Process]] — the setting.
- [[Bellman Equations]] — what VI iterates.
- [[Policy Iteration]] — alternative algorithm.
- [[VI Generative Setting]] — sample-complexity analysis when $\mathbb{P}$ is unknown.
- [[Q-learning]] — sample-based, online version (one Bellman update per transition, no need for full $\mathbb{P}$).
