# Optimism Principle (Optimism in the Face of Uncertainty)

**Tags:** #foundational #framework #bandits

**Generic recipe** for exploration under uncertainty: act as if the most plausible *favorable* hypothesis were true. The conceptual backbone of UCB and its many descendants.

## The recipe (two steps)

**Step 1.** Build a **set of statistically plausible models**: for each arm $a$ at round $t$, construct a confidence interval $[\mathrm{LCB}_a(t),\, \mathrm{UCB}_a(t)]$ around the unknown parameter (e.g. the mean $\mu_a$).

**Step 2.** Act as if the **best plausible model** were the true one — i.e. play the action that maximizes the value under the most favorable model in the confidence set:
$$a_t = \arg\max_a \mathrm{UCB}_a(t) \;=\; \arg\max_{a,\,\mu_a \in [\mathrm{LCB}_a, \mathrm{UCB}_a]}\;\mu_a.$$

The "double max" — over actions *and* over plausible parameters — is what gives the principle its name.

## Why optimism gives both exploration and exploitation

- **High empirical mean → high UCB → exploited.** The algorithm prefers arms that have looked good so far.
- **Few samples → wide confidence interval → high UCB → explored.** Arms that haven't been pulled enough have inflated UCBs, forcing the algorithm to try them.

As an arm is pulled more, its UCB shrinks (radius $\propto 1/\sqrt{N_t(a)}$). Eventually, only arms with truly high means stay competitive on UCB.

## Why optimism (rather than e.g. realism) works

The proof structure is **always the same** (see [[Clean Event]] / [[UCB Analysis]]):

1. On a high-probability "[[Clean Event|clean event]]" $\mathcal{G}$, $\mathrm{UCB}_a(t) \geq \mu_a$ for all $(a, t)$.
2. In particular $\mathrm{UCB}_{a^*}(t) \geq \mu^* = \mu_{a^*}$.
3. If the algorithm picks a suboptimal $a$ at time $t$, then $\mathrm{UCB}_a(t) \geq \mathrm{UCB}_{a^*}(t) \geq \mu^*$.
4. But $\mathrm{UCB}_a(t)$ is also upper-bounded by $\mu_a + 2 \cdot \text{radius}$ on $\mathcal{G}$.
5. Combining: $\mu^* \leq \mu_a + 2 \cdot \text{radius}_a$, i.e. $\Delta_a \leq 2 \cdot \text{radius}_a$. This **bounds the gap** an arm can sustain while still being pulled — which bounds $N_t(a)$ → bounds regret.

Pessimism (LCB) would do the opposite: avoid arms with low LCB, lock onto whichever arm has the best lower bound. This is good for **safety** but terrible for **regret minimization** — it severely under-explores.

## Universality conditions (Celli, slide 15)

The optimism principle yields sublinear-regret algorithms when:

1. **Every action gives feedback about its own quality.**  *(Pulling arm $a$ reveals samples of $\mu_a$.)*
2. **No action gives feedback about the value of other actions.**  *(Pulling arm $a$ tells you nothing about $\mu_b$ for $b \neq a$.)*

### What happens when (1) is violated

If some action's quality cannot be learned from pulling it (e.g. delayed feedback, censored observations, partial monitoring), **sublinear regret may be impossible** with optimism alone. You may need more sophisticated machinery (e.g. partial-monitoring games).

### What happens when (2) is violated

If pulling arm $a$ also reveals information about $\mu_b$, optimism may **miss informative-but-low-reward actions**. Example: in a combinatorial bandit where actions are subsets, pulling subset $\{1, 2\}$ reveals info about both items, but pulling singleton $\{1\}$ gives information at lower reward. Pure optimism plays the high-UCB subsets and never explores cheap-but-informative singletons.

In such settings, alternative strategies (e.g. **information-directed sampling**, **explore-then-exploit**, or **forced exploration**) outperform optimism.

## Instances in this vault

| Setting | Algorithm | Confidence set built from |
|---|---|---|
| [[Stochastic Bandits]] | [[UCB1]] | Hoeffding radius $\sqrt{2 \log T / N_t(a)}$ |
| | KL-UCB | KL-divergence ball |
| [[Contextual Bandit]] (linear) | [[LinUCB]] | Ellipsoid $x_t^\top A_t^{-1} x_t$ |
| RL (finite-horizon MDP) | UCB-VI, UCRL2 | Hoeffding on $P, R$ + value-iteration plan |
| Continuous bandits | GP-UCB | Gaussian-process posterior |

## Alternatives

- **[[Thompson Sampling]]** — randomization-based; samples from posterior instead of using confidence sets. Often empirically better, similar theoretical guarantees.
- **Information-directed sampling** (Russo-Van Roy) — explicitly balances expected regret vs. information gain. Strictly more general than optimism.
- **Boltzmann / softmax exploration** — heuristic; no general regret theory.

## See also

- [[UCB1]] — the prototypical optimism algorithm.
- [[LinUCB]] — optimism for linear contextual bandits.
- [[Clean Event]] — the analysis tool optimism proofs all use.
- [[Thompson Sampling]] — the Bayesian sibling.
