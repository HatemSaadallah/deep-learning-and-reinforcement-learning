# Follow the Perturbed Leader (FTPL)

**Tags:** #online-learning #algorithm

Stabilizes [[Follow the Leader]] by adding random noise to cumulative losses.

## Rule

Sample noise $z \sim \mathcal{D}$, then play
$$x_t = \arg\min_{x \in \mathcal{K}} \left[ \sum_{s=1}^{t-1} f_s(x) + z \cdot x \right].$$

## Why noise helps

Stochastic tie-breaking prevents the wild swings that hurt vanilla FTL.

## Use in RL

- [[Adversarial Bandits]] — Exp3-style algorithms can be derived from FTPL with exponential noise.

## See also

- [[Follow the Regularized Leader]] — deterministic alternative.
