# Online Convex Optimization (OCO)

**Tags:** #online-learning #framework #foundational

The most-studied subfield of [[Online Learning]]. Restricts the general protocol to **convex** decision sets and **convex** loss functions, which makes regret minimization analytically tractable.

## Framework (Zinkevich 2003)

Let $\mathcal{C} \subseteq \mathbb{R}^d$ be a convex set. At each round $t = 1, 2, \dots, T$:
1. Learner picks $\theta_t \in \mathcal{C}$.
2. Environment reveals a **convex** loss function $\ell_t: \mathcal{C} \to \mathbb{R}$.
3. Learner suffers $\ell_t(\theta_t)$.

**Regret:**
$$R_T \;=\; \sum_{t=1}^T \ell_t(\theta_t) \;-\; \min_{\theta \in \mathcal{C}} \sum_{t=1}^T \ell_t(\theta).$$

(Compare to best **single static choice** in hindsight.)

## Why convex?

- Convex losses admit subgradients with the property $\ell_t(\theta) \geq \ell_t(\theta_t) + \langle g_t, \theta - \theta_t\rangle$. This linearizes regret: bounding $\sum \langle g_t, \theta_t - \theta^*\rangle$ suffices.
- Convex feasible set means projections $\Pi_\mathcal{C}$ are well-defined and contract distances to feasible points.

## Canonical algorithms in this vault

| Algorithm | Update | Regret | Note |
|---|---|---|---|
| [[Follow the Leader]] | best on past losses | $\Omega(T)$ (unstable) | fails on linear losses |
| **OGD** ([[OGD Regret Bound|proof]]) | $\theta_{t+1} = \Pi_\mathcal{C}(\theta_t - \eta g_t)$ | $RL\sqrt{T}$ | Zinkevich's first algorithm |
| [[Follow the Regularized Leader|FTRL]] | $\theta_{t+1} = \arg\min \eta\!\!\sum\ell_s + R$ | $O(\sqrt{T})$ | adds regularizer to FTL |
| [[Mirror Descent Analysis|Mirror Descent]] | $\theta_{t+1} = \arg\min \eta\langle g_t, \theta\rangle + D_R(\theta, \theta_t)$ | $O(\sqrt{T})$ | uses [[Bregman Divergence|Bregman]] geometry |
| ONS (Online Newton Step) | second-order update | $O((d/\alpha) \log T)$ | for [[Exp-Concavity|exp-concave]] losses |

## Regret rates by problem structure

| Loss structure | Best achievable regret | Algorithm |
|---|---|---|
| Convex + Lipschitz | $\Theta(\sqrt{T})$ | OGD, FTRL, MD |
| Strongly convex | $\Theta(\log T)$ | OGD with $\eta_t = 1/(\mu t)$, FTRL |
| [[Exp-Concavity\|Exp-concave]] | $\Theta(d \log T)$ | ONS, [[Online Ridge Regression\|online ridge regression]] |
| Smooth + convex | $\Theta(\sqrt{T})$, but $O(L \log T)$ if optimal value is small | OGD |

## Instances inside the course

- **[[Online Ridge Regression]]** — OCO with squared loss; $O(\log T)$ regret via exp-concavity.
- **[[Online Logistic Regression]]** — OCO with logistic loss; the running example in Celli's intro lecture.
- **[[Hedge - Multiplicative Weights]]** — OCO on the simplex with linear losses.
- **[[Bandit Lower Bound|Bandit problems]]** — partial-information OCO (only $\ell_t(\theta_t)$ is observed, not $\ell_t(\cdot)$).

## References

- Zinkevich, *Online convex programming and generalized infinitesimal gradient ascent*, ICML 2003.
- Hazan, *Introduction to online convex optimization*, MIT Press 2022.
- Orabona, *A modern introduction to online learning*, arXiv:1912.13213, 2019.

## See also

- [[Online Learning]] — the parent framework.
- [[Regret]] — the performance measure.
- [[OGD Regret Bound]] — the canonical proof in this framework.
