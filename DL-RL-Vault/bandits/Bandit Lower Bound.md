# Bandit Lower Bound: $\Omega(\sqrt{KT})$

**Tags:** #bandits #lower-bound #theorem

## Theorem

Fix $T$ and $K$. For **any** bandit algorithm (deterministic or randomized), there exists an instance $\mathcal{D}$ such that
$$R^T_\mathcal{D} \geq \Omega(\sqrt{KT}).$$

This is **worst-case / instance-independent**. A given algorithm can do much better on specific instances (e.g. when gaps $\Delta_a$ are large).

## Proof outline

1. **Hard family** $\mathcal{I}_j$ from [[Best-Arm Identification]]: one arm at $1/2 + \epsilon/2$, rest at $1/2$.
2. **Information-theoretic indistinguishability:** under any algorithm, $T \leq cK/\epsilon^2$ rounds are not enough to identify the good arm with prob $> 3/4$.
   - [[KL Divergence]] chain rule + [[Pinskers Inequality]] bound how much the algorithm's behavior differs across instances.
3. **[[Yaos Principle]]:** extends from deterministic to randomized algorithms.
4. **Identification → regret:** if you pick the wrong arm with constant probability, you accumulate $\Omega(\epsilon T)$ regret on the missed instance.
5. **Balance:** set $\epsilon = \sqrt{K/T}$ → regret $\Omega(\sqrt{KT})$.

## Comparison with [[Explore-Then-Commit|ETC]]

| | Regret |
|---|---|
| ETC (upper) | $\widetilde{O}(K^{1/3} T^{2/3})$ |
| **Lower bound** | $\Omega(\sqrt{KT})$ |
| UCB (matches LB) | $\widetilde{O}(\sqrt{KT})$ |

ETC has the **wrong exponent** in $T$. UCB closes the gap.
