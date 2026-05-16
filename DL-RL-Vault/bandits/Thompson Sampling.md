# Thompson Sampling

**Tags:** #bandits #algorithm #bayesian

A **Bayesian** approach to [[Stochastic Bandits|stochastic bandits]]: maintain a posterior over each arm's mean, sample from it, act greedy on the sample. Exploration emerges from **randomization** rather than confidence bounds.

Historically the first formal bandit algorithm — Thompson 1933 — predating UCB by ~70 years.

## Algorithm (generic)

Maintain a prior $\pi_0(a)$ over each arm $a$'s reward parameter $\theta_a$.

**For** $t = 1, 2, \dots$:
1. **Sample** $\tilde\theta_a \sim \pi_{t-1}(a)$ for every $a$.
2. **Play** $a_t = \arg\max_a \mathbb{E}_{r \sim \tilde\theta_a}[r]$.
3. **Observe** $r_t \sim \mathcal{D}_{a_t}$.
4. **Update** posterior $\pi_t(a_t) \propto \pi_{t-1}(a_t) \cdot p(r_t \mid \tilde\theta_{a_t})$. Other arms' posteriors unchanged.

The optimism in UCB is replaced by **noise injection from posterior sampling**: poorly-known arms have wide posteriors, so their samples vary a lot, giving them a fair chance to win the argmax.

## Bernoulli case (the canonical example)

- **Prior**: $\mathrm{Beta}(\alpha_0, \beta_0)$ for each arm (often $\alpha_0 = \beta_0 = 1$, i.e. uniform on $[0,1]$).
- **Posterior** after $s$ successes and $f$ failures: $\mathrm{Beta}(\alpha_0 + s,\; \beta_0 + f)$.
- **Update on reward $r \in \{0, 1\}$**: $\alpha \mathrel{+}= r$, $\beta \mathrel{+}= (1 - r)$.

The whole algorithm:
```
for each round t:
    for each arm a: sample theta_a ~ Beta(alpha_a, beta_a)
    play a_t = argmax_a theta_a
    observe r_t
    alpha_{a_t} += r_t; beta_{a_t} += 1 - r_t
```

Implemented in `code/01_bandits/agents.py` as `ThompsonSamplingBeta`. In the bandit experiment it **decisively beats UCB1** on the test instance (final regret 82 vs. 368).

## Gaussian case

- **Prior**: $\mathcal{N}(\mu_0, \sigma_0^2)$ per arm.
- **Posterior** after $n$ samples with empirical mean $\hat\mu$: $\mathcal{N}\Bigl(\frac{\sigma_0^{-2}\mu_0 + n \sigma^{-2} \hat\mu}{\sigma_0^{-2} + n\sigma^{-2}},\; \frac{1}{\sigma_0^{-2} + n\sigma^{-2}}\Bigr)$.

## Guarantees

| Bound | Rate |
|---|---|
| Frequentist regret (Agrawal-Goyal 2012, Kaufmann-Korda-Munos 2012) | $\widetilde{O}(\sqrt{KT})$ |
| Bayesian regret (Russo-Van Roy 2014) | $O(\sqrt{KT \log T})$ |
| Instance-dependent (Bernoulli) | $O(\log T \sum_a 1/\Delta_a)$, with the **optimal constant** (matches the [[Bandit Lower Bound]]) |

The Bernoulli regret bound matches KL-UCB's optimal constant — TS is **not just empirical**; it's also asymptotically optimal in theory.

## Why exploration via randomization works

If arm $a$'s posterior is poorly concentrated (i.e. few samples), then $\tilde\theta_a$ samples can be high *some of the time*. The probability that a poorly-sampled arm "wins" the argmax over a well-sampled arm is exactly its **posterior probability of being best** — which UCB's confidence bound approximates from above, but TS computes (samples from) directly.

This makes TS:
- **Self-calibrating**: as the posterior tightens, sampling variance shrinks, exploration decreases.
- **Hyperparameter-free**: no exploration constant to tune (unlike $\epsilon$-greedy).
- **Often empirically better** than UCB: less over-exploration when the optimal arm is clearly identified.

## When the optimism principle fails ([[Optimism Principle]] condition violations)

Thompson Sampling **also requires** the same two conditions as UCB:
1. Every action gives feedback about itself.
2. No action gives feedback about others.

So TS shares UCB's failure modes in combinatorial / partial-monitoring settings. The advantage of TS over UCB lies in the constants and ease of generalization, not in handling fundamentally different problem structures.

## Generalizations

- **Linear TS** (Agrawal-Goyal 2013) — Gaussian posterior over $\theta_a \in \mathbb{R}^d$ for [[Contextual Bandit|linear contextual bandits]]. Often beats [[LinUCB]] in practice.
- **Neural TS** — replace linear with NN, use dropout / ensembles / Bayesian last layer for posterior approximation.
- **PSRL** (Posterior Sampling for Reinforcement Learning, Osband et al.) — the natural TS extension to MDPs. Sample an MDP from the posterior, plan optimally in it.

## When NOT to use TS

- **No good prior available.** TS needs at least a vague prior; misspecified priors can hurt.
- **Computationally heavy posteriors.** For complex models, posterior sampling may be expensive (vs. UCB which only needs point estimates + confidence radii).

## References

- Thompson, *On the likelihood that one unknown probability exceeds another in view of the evidence of two samples*, Biometrika 1933 (!).
- Russo, Van Roy, Kazerouni, Osband, Wen, *A Tutorial on Thompson Sampling*, FnT in ML 2018.
- Lattimore & Szepesvári, *Bandit Algorithms* (2020), Ch. 36.

## See also

- [[UCB1]] — the frequentist alternative.
- [[Optimism Principle]] — both UCB and TS satisfy its conditions.
- [[Stochastic Bandits]] — parent setting.
