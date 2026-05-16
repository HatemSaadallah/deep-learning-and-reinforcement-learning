# Sub-optimality Gap

**Tags:** #bandits #definition

For a [[Stochastic Bandits|stochastic bandit]] instance:
$$\Delta_a := \mu^* - \mu_a \geq 0.$$

Measures how much you lose *in expectation per pull* by playing arm $a$ instead of the optimal arm.

## Why it matters

- Appears directly in [[Regret Decomposition Lemma]]: $R^T = \sum_a \Delta_a \mathbb{E}[N_a(T)]$.
- **Instance-dependent regret bounds** scale with $1/\Delta_a$ (small gaps are harder).
- **Worst-case regret bounds** (like the [[Bandit Lower Bound|$\sqrt{KT}$ LB]]) are obtained by picking $\Delta_a$ adversarially small ($\epsilon \sim 1/\sqrt{T}$).
