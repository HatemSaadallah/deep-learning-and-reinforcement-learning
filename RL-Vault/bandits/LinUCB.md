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

## Where does the confidence ellipsoid come from? (Celli's derivation, Lecture 7)

The full setting is **stochastic linear bandits**: at each round, an action set $A_t \subset \mathbb{R}^d$ is revealed, the learner picks $a_t \in A_t$, and observes $\ell_t = \langle a_t, \theta^*\rangle + \epsilon_t$ with $\epsilon_t \sim \mathcal{N}(0, 1)$ independent.

**Step 1 — Center via ridge regression.** Define the Gram matrix and pseudo-target:
$$M_t = \sum_{\tau \leq t} a_\tau a_\tau^\top, \qquad Z_t = \sum_{\tau \leq t} \epsilon_\tau a_\tau.$$
Then $\hat\theta_t := M_t^{-1} \sum \ell_\tau a_\tau$ satisfies $\hat\theta_t - \theta^* = M_t^{-1} Z_t$.

**Step 2 — Distribution of the error.** Conditional on a fixed-design action sequence (i.e. $a_\tau$ chosen before seeing $\epsilon_\tau$):
$$Z_t \sim \mathcal{N}(0,\, M_t) \;\Longrightarrow\; M_t^{1/2}(\hat\theta_t - \theta^*) = M_t^{-1/2} Z_t \sim \mathcal{N}(0,\, I_d).$$

**Step 3 — Chi-squared confidence.** The squared $\ell_2$ norm:
$$\|M_t^{1/2}(\hat\theta_t - \theta^*)\|_2^2 = \|\hat\theta_t - \theta^*\|_{M_t}^2 \sim \chi^2_d.$$
By chi-squared tail bounds, with probability $\geq 1 - \delta$:
$$\|\hat\theta_t - \theta^*\|_{M_t}^2 \;\leq\; d + 2\sqrt{d \log(1/\delta)} + 2 \log(1/\delta).$$

**Confidence set:** $\Theta_t = \bigl\{ \theta : \|\theta - \hat\theta_t\|_{M_t} \leq \beta_t\bigr\}$ — an **ellipsoid** centered at $\hat\theta_t$, with axes determined by eigenvectors of $M_t$.

**Step 4 — Dropping fixed-design assumption (Abbasi-Yadkori et al. 2011).** The argument above assumes $a_t$ chosen before $\epsilon_t$, which is **not true** in practice (LinUCB selects $a_t$ based on past data). The fix uses **self-normalized concentration**:
$$\beta_t = \lambda^{1/2} + \sqrt{2 \log(1/\delta) + d \log(1 + t/(d\lambda))}.$$
With this $\beta_t$, $\theta^* \in \Theta_t$ for **all $t$** with probability $\geq 1 - \delta$, regardless of how $a_t$ depends on history. This is the result that makes LinUCB rigorous.

**Step 5 — Algorithmic form.** Given $\Theta_t$, compute the LCB (or equivalently UCB):
$$\mathrm{LCB}_{t+1}(a) = \min_{\theta \in \Theta_t} \langle a, \theta\rangle = \langle a, \hat\theta_t\rangle - \beta_t \|a\|_{M_t^{-1}}.$$
LinUCB plays $a_{t+1} = \arg\min_a \mathrm{LCB}_{t+1}(a)$ for losses (equivalently $\arg\max$ for rewards).

The bonus $\|a\|_{M_t^{-1}}$ measures **how poorly aligned $a$ is with directions seen in past data** — large for under-explored directions, small for well-sampled ones.

## How $a^\top M_t^{-1} a$ behaves visually

- **Few samples in random directions:** ellipsoid is large and round → high exploration bonus everywhere.
- **Many samples, all in one direction $e_1$:** ellipsoid is narrow along $e_1$ but wide along $e_2, \dots, e_d$ → bonus is small for $e_1$-aligned actions, huge for orthogonal ones.
- **Asymptotic shrinkage:** as $T \to \infty$ with non-degenerate exploration, $M_t \sim T \cdot \Sigma$ (covariance) and the ellipsoid shrinks at rate $1/\sqrt{T}$.

## Practical notes

- **Online update via Sherman-Morrison:** $A_a^{-1}$ can be updated in $O(d^2)$ per round without re-inverting.
- **Disjoint vs hybrid:** "disjoint" keeps a separate $\theta_a$ per arm. The Li et al. paper also defines a "hybrid" version with shared features across arms — better when arms share structure.
- $\alpha$ is a hyperparameter (theory says $\alpha \sim \sqrt{\log(T/\delta)}$, practice tunes it).

## Alternatives

- **Contextual Thompson Sampling** (Agrawal & Goyal): often beats LinUCB empirically. Posterior $\theta_a \sim \mathcal{N}(\hat\theta_a, \sigma^2 A_a^{-1})$, sample, act greedily.
- **Neural LinUCB / Deep Bayesian bandits**: NN feature extractor + linear head + LinUCB on the head.
