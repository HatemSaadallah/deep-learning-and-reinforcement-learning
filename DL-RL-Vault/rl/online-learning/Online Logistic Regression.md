# Online Logistic Regression

**Tags:** #online-learning #algorithm

The running example the lecture uses in the intro lecture (slides 14, 19, 22) to motivate why we need a *generic* [[Online Convex Optimization|OCO]] framework: unlike [[Online Ridge Regression|ridge]], logistic regression has **no closed-form solution**, so there's no analog of Sherman-Morrison to fall back on.

## Setting

Features $x \in \mathbb{R}^d$, labels $y \in \{-1, +1\}$. Linear predictor parameterized by $\theta \in \mathbb{R}^d$. Logistic loss:
$$\ell(y, \theta; x) \;:=\; \ln\!\bigl(1 + \exp(-y \langle \theta, x\rangle)\bigr).$$

The loss is **convex** in $\theta$, **smooth**, and on bounded domains it is also [[Exp-Concavity|$\alpha$-exp-concave]] for some $\alpha$ that depends exponentially on the domain radius.

## Online protocol

At each round $t = 1, \dots, T$:
1. Choose parameter vector $\theta_t \in \mathbb{R}^d$.
2. Observe loss function $\ell_t(\theta) = \ln(1 + \exp(-y_t \langle \theta, x_t\rangle))$.
3. Suffer $\ell_t(\theta_t)$.

**Regret:**
$$R_T \;=\; \sum_{t=1}^T \ell_t(\theta_t) \;-\; \min_{\theta \in \mathbb{R}^d} \sum_{t=1}^T \ell_t(\theta).$$

The min-over-$\theta$ on the RHS is the **batch logistic regression** estimator — i.e. regret compares the online learner to the offline maximum-likelihood predictor trained on all data in hindsight.

## Why it's harder than ridge

Offline logistic regression solves
$$\hat\theta_n \in \arg\min_\theta \sum_{t=1}^n \ln(1 + \exp(-y_t \langle x_t, \theta\rangle))$$
which **has no closed-form** — must be solved iteratively (Newton, IRLS, gradient methods). So there's no Sherman-Morrison-style trick to maintain $\hat\theta$ recursively.

→ **The fix is to give up on tracking the offline ERM exactly**, and instead use an online algorithm with a regret guarantee against it.

## Algorithms and their regret

| Algorithm | Per-round update | Regret | Comments |
|---|---|---|---|
| **OGD** | $\theta_{t+1} = \theta_t - \eta \nabla \ell_t(\theta_t)$ | $O(RG\sqrt{T})$ | Generic OCO, ignores exp-concavity |
| **Online Newton Step** | $\theta_{t+1} = \Pi^{A_t}(\theta_t - \tfrac{1}{\alpha} A_t^{-1} g_t)$ | $O((d/\alpha) \log T)$ | Exploits [[Exp-Concavity\|exp-concavity]]; can have very bad $\alpha$ |
| **FTRL with KL or Euclidean regularizer** | minimize cumulative loss + regularizer | $O(R G \sqrt{T})$ | Equivalent rate to OGD |

The OGD gradient is closed-form:
$$\nabla \ell_t(\theta) \;=\; -\frac{y_t x_t}{1 + \exp(y_t \langle \theta, x_t\rangle)} \;=\; -y_t\, \sigma(-y_t \langle \theta, x_t\rangle)\, x_t$$
where $\sigma$ is the sigmoid. Note $\|\nabla \ell_t\| \leq \|x_t\|$, giving a clean Lipschitz constant.

## Why exp-concavity is delicate here

Logistic loss **is** exp-concave on bounded domains, but the constant $\alpha$ degrades **exponentially** in the domain radius $R$: $\alpha = \Theta(e^{-R \cdot X})$ for $\|\theta\| \leq R$, $\|x\| \leq X$.

So while ONS gives "logarithmic" regret in $T$, the constant in front blows up dramatically as the domain grows. For high-dimensional or large-norm settings, the $\sqrt{T}$ bound of OGD is often preferred in practice.

This is a recurring theme in OCO: **theoretical rates and practical performance diverge** when problem constants matter.

## Why it's the canonical OCO example in the course

the intro lecture (slide 14, 19, 22) sets up logistic regression as:
- A **familiar** ML problem (most students have done logistic regression offline).
- One where the **batch closed form fails** (unlike linear regression).
- A **clean convex loss** to which generic OCO algorithms apply.
- A concrete instance of the regret framework $R_T = \sum \ell_t(\theta_t) - \min_\theta \sum \ell_t(\theta)$.

It motivates everything that follows: OGD (slide 25), the [[OGD Regret Bound|$\sqrt{T}$ regret theorem]] (slide 27), and the eventual generalization to [[Mirror Descent Analysis|mirror descent]] and [[Follow the Regularized Leader|FTRL]].

## See also

- [[Online Convex Optimization]] — the framework.
- [[Online Ridge Regression]] — the easier sibling (closed form + log-$T$ regret).
- [[Exp-Concavity]] — what limits the regret here.
- [[Convexity]] — properties of the logistic loss.
