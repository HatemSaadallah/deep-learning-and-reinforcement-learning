# Best-Arm Identification

**Tags:** #bandits #lower-bound

Variant of [[Stochastic Bandits]] where the objective is **identification**, not cumulative reward: after $T$ rounds, output a single prediction $y_T$ of the optimal arm.

## Lower bound (sample complexity)

For **any** algorithm to achieve $\Pr[y_T = a^*] \geq 1 - \text{small}$ on every instance, it needs
$$T \geq \Omega(K/\epsilon^2)$$
where $\epsilon$ characterizes the smallest gap.

## Hard instance family

For $j \in [K]$, define instance $\mathcal{I}_j$ with Bernoulli arms:
$$\mu_i = \begin{cases} 1/2 + \epsilon/2 & i = j \\ 1/2 & i \neq j \end{cases}$$

The $K$ instances are pairwise hard to distinguish — they differ in only one arm by $\epsilon/2$.

## Lemma (key step)

For $T \leq cK/\epsilon^2$ (small constant $c$), any deterministic algorithm fails on $\geq \lceil K/3 \rceil$ of the instances:
$$\Pr[y_T = a \mid \mathcal{I}_a] < 3/4 \text{ for at least } \lceil K/3 \rceil \text{ arms } a.$$

## Proof sketch (two arms)

By contradiction: suppose $\Pr_1[A] \geq 3/4$ and $\Pr_2[A] \leq 1/4$ for $A = \{y_T = 1\}$, so $P_1(A) - P_2(A) \geq 1/2$.

Apply [[Pinskers Inequality]] + chain rule on the reward-table distribution:
$$2(P_1(A) - P_2(A))^2 \leq \mathrm{KL}(P_1, P_2) \leq 2T \cdot \mathrm{KL}(\mathrm{RC}_\epsilon, \mathrm{RC}_0) \leq 2T \cdot 2\epsilon^2.$$

So $P_1(A) - P_2(A) \leq \epsilon\sqrt{2T} < 1/2$ when $T \leq 1/(4\epsilon)^2$. **Contradiction.**

## From identification → regret

Choose the instance uniformly at random, then $\Pr[y_T \neq a^*] \geq 1/12$. [[Yaos Principle]] lifts this to randomized algorithms. Converting identification failure to cumulative regret gives the [[Bandit Lower Bound|$\Omega(\sqrt{KT})$ regret bound]].
