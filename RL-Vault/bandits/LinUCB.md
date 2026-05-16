# LinUCB

**Tags:** #bandits #algorithm

UCB1 generalized to the linear [[Contextual Bandit]] setting.

## Model

Assume $\mu(x, a) = \langle \theta_a, x \rangle$ for unknown $\theta_a \in \mathbb{R}^d$.

## Per-arm sufficient statistics

For each arm $a$, maintain:
- $A_a = \lambda I + \sum_{s : a_s = a} x_s x_s^\top \in \mathbb{R}^{d \times d}$ (regularized Gram matrix)
- $b_a = \sum_{s : a_s = a} r_s x_s \in \mathbb{R}^d$

Ridge-regression estimate:
$$\hat\theta_a = A_a^{-1} b_a.$$

## Selection rule

$$a_t = \arg\max_a \; \underbrace{\hat\theta_a^\top x_t}_{\text{exploitation}} \;+\; \alpha \underbrace{\sqrt{x_t^\top A_a^{-1} x_t}}_{\text{confidence radius}}$$

The second term is the **width of the confidence ellipsoid for $\hat\theta_a^\top x_t$** — large when $x_t$ points in a direction with few observations for arm $a$, small when arm $a$ has seen many similar contexts.

Compare to [[Stochastic Bandits|UCB1]]'s bonus $\sqrt{2 \log t / N_a(t)}$ — same exploration-exploitation logic, but the "count" is replaced by directional coverage via $A_a^{-1}$.

## Regret guarantee

$$R^T = \widetilde{O}(d\sqrt{T})$$

Crucially, **independent of $|\mathcal{A}|$** — only the feature dimension $d$ matters. This is what makes contextual bandits scale to huge or infinite action sets when actions share structure.

## Practical notes

- **Online update via Sherman-Morrison:** $A_a^{-1}$ can be updated in $O(d^2)$ per round without re-inverting.
- **Disjoint vs hybrid:** "disjoint" keeps a separate $\theta_a$ per arm. The Li et al. paper also defines a "hybrid" version with shared features across arms — better when arms share structure.
- $\alpha$ is a hyperparameter (theory says $\alpha \sim \sqrt{\log(T/\delta)}$, practice tunes it).

## Alternatives

- **Contextual Thompson Sampling** (Agrawal & Goyal): often beats LinUCB empirically. Posterior $\theta_a \sim \mathcal{N}(\hat\theta_a, \sigma^2 A_a^{-1})$, sample, act greedily.
- **Neural LinUCB / Deep Bayesian bandits**: NN feature extractor + linear head + LinUCB on the head.
