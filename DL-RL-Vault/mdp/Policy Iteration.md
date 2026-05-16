# Policy Iteration (PI)

**Tags:** #mdp #algorithm #planning

Alternate **policy evaluation** (compute $Q^\pi$) and **policy improvement** (greedy w.r.t. $Q^\pi$) until convergence. Often converges in **fewer iterations** than [[Value Iteration]], at the cost of a more expensive per-iteration step.

## Algorithm

**Input:** MDP $(\mathcal{S}, \mathcal{A}, r, \mathbb{P}, T)$ (or discounted version).

**Initialize:** arbitrary policies $\pi^0 = \{\pi^0_h\}_{h=1}^T$.

**For** $k = 1, 2, \dots$:
1. **Policy evaluation:** compute $Q^{\pi^{k-1}}_h(s, a)$ for all $(s, a, h)$ using [[Bellman Equations|Bellman evaluation]]:
$$Q^{\pi^{k-1}}_h(s, a) = r_h(s, a) + (\mathbb{P}_h V^{\pi^{k-1}}_{h+1})(s, a).$$
2. **Policy improvement:** define $\pi^k_h(s) := \arg\max_a Q^{\pi^{k-1}}_h(s, a)$ for all $(s, h)$.

**Stop** when $\pi^k = \pi^{k-1}$.

## Convergence

For **finite-horizon** MDPs, $\pi^k$ becomes optimal in at most $T$ iterations (one per backward stage). After that, $\pi^k = \pi^*$.

For **discounted infinite-horizon** MDPs, PI converges in finite time (it visits a non-repeating sequence of deterministic policies, of which there are finitely many: $|\mathcal{A}|^{|\mathcal{S}|}$). In practice, **far** fewer iterations than the worst case — often $O(\log(1/(1-\gamma)))$.

## Policy improvement theorem

The improvement step is **monotone**: $V^{\pi^k}(s) \geq V^{\pi^{k-1}}(s)$ for all $s$, with strict inequality somewhere unless $\pi^{k-1}$ is already optimal.

**Proof sketch:** by greedy choice, $\sum_a \pi^k(a|s) Q^{\pi^{k-1}}(s, a) \geq \sum_a \pi^{k-1}(a|s) Q^{\pi^{k-1}}(s, a) = V^{\pi^{k-1}}(s)$. Apply this in Bellman's equation for $V^{\pi^k}$ and iterate.

This monotone-improvement property generalizes to **[[TRPO Surrogate Objective|TRPO]]** in the function-approximation setting (where exact policy evaluation is replaced by approximations).

## Cost comparison with VI

| | Per iteration | # iterations |
|---|---|---|
| **[[Value Iteration]]** | $O(\|\mathcal{S}\|^2 \|\mathcal{A}\|)$ Bellman backup | $O(\tfrac{1}{1-\gamma} \log \tfrac{1}{\epsilon})$ |
| **PI (exact eval)** | $O(\|\mathcal{S}\|^3)$ matrix solve + $O(\|\mathcal{S}\|^2 \|\mathcal{A}\|)$ greedy | Finite (often small) |
| **Modified PI** | $O(K \|\mathcal{S}\|^2 \|\mathcal{A}\|)$ — $K$ Bellman backups for eval | Between VI and exact PI |

The exact policy evaluation step requires solving a linear system $V^\pi = r + \mathbb{P}^\pi V^\pi$, i.e. $(I - \mathbb{P}^\pi) V^\pi = r$. For large state spaces, this is intractable — hence **modified PI** truncates the evaluation to a few Bellman backups (so it's basically a hybrid with VI).

## See also

- [[Markov Decision Process]] — the setting.
- [[Bellman Equations]] — the evaluation step.
- [[Value Iteration]] — alternative algorithm with simpler per-iteration cost.
- [[TRPO Surrogate Objective]] — generalizes the monotone-improvement idea to function approximation.
