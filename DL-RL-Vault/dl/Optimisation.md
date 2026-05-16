# Optimisation (Deep Learning)

**Tags:** #dl #optimisation #foundational

How to **fit** the parameters $\phi$ of a neural network — minimize the empirical loss
$$L[\phi] = \frac{1}{N}\sum_{i=1}^N \ell_i\bigl(f(x_i; \phi),\, y_i\bigr).$$

Unlike classical convex optimisation, the loss landscape of a deep net is **non-convex**, **non-smooth**, and **astronomically high-dimensional** ($\phi \in \mathbb{R}^D$ with $D$ in the millions to billions).

## Landscape vocabulary

- **Local minimum:** $L(\phi^*) \leq L(\phi)$ for $\phi$ in a neighborhood.
- **Global minimum:** $L(\phi^*) \leq L(\phi)$ everywhere.
- **Saddle point:** $\nabla L = 0$, but Hessian has both positive and negative eigenvalues — increases in some directions, decreases in others.
- **Critical points:** any point with $\nabla L = 0$ (minima, maxima, saddles).
- **Basin of attraction:** set of $\phi_0$ that converge to a given minimum under gradient descent.
- **Test for convexity:** Hessian $H[\phi]$ positive semi-definite everywhere (all eigenvalues $\geq 0$). NN loss landscapes **fail** this test almost everywhere.

## Two empirical surprises of deep learning loss landscapes

1. **Saddle points are far more common than local minima** in high dimensions. Random matrix theory: for a Hessian with random eigenvalue signs, the probability that all $D$ are positive (a minimum) is $\sim 2^{-D}$ — vanishing for big $D$. → Most $\nabla L = 0$ points are saddles, not minima.
2. **Skip connections (ResNet) smooth out the landscape dramatically.** ResNet-56 without skip connections has a wildly bumpy loss surface; *with* them it becomes nearly convex around the minimum (Li et al. 2018, "Visualizing the Loss Landscape").

## Gradient flow → gradient descent (discretisation)

**Continuous-time view:** negative gradient flow
$$\dot \phi = -\nabla_\phi L \quad\Longrightarrow\quad \frac{dL}{dt} = -\|\nabla_\phi L\|^2 \leq 0.$$
$L$ decreases monotonically along trajectories.

**Discretisation (Euler integration)** with step size $h$ gives **gradient descent**:
$$\phi^t = \phi^{t-1} - h \nabla_\phi L(\phi^{t-1}).$$

The discrete dynamics can:
- **Converge** (small enough $h$),
- be **unstable** (oscillations, moderate $h$),
- **diverge** (large $h$).

For quadratic objectives the stability threshold is known: $h < 2/L_{\max}$ where $L_{\max}$ is the largest Hessian eigenvalue. For NNs it's far more complex and the safe $h$ has to be found empirically.

## Full-batch, mini-batch, stochastic GD

| | Per-step gradient | Memory | Noise |
|---|---|---|---|
| **Full-batch GD** | $\nabla L$ over all $N$ examples | high | none |
| **Mini-batch SGD** | $\nabla \hat L$ over $B \ll N$ examples | low | moderate |
| **Stochastic GD** | $\nabla \ell_i$ on a single example | minimal | high |

**Mini-batch SGD** is the workhorse — its gradient noise acts as **implicit regularisation** (favors flat minima which generalize better, see [[Overparameterisation]]). $B$ typically 32–512.

## Momentum

Plain SGD oscillates in ravine-like regions. **Momentum** accumulates a velocity:
$$v_t = \mu\, v_{t-1} + \nabla L(\phi_{t-1}), \qquad \phi_t = \phi_{t-1} - \eta\, v_t.$$
- $\mu \in [0, 1)$ (typically 0.9): friction coefficient.
- Damps oscillations across the curvature, accelerates along low-curvature directions.

**Nesterov's accelerated gradient (NAG)** evaluates the gradient at the look-ahead point: $\nabla L(\phi_{t-1} - \eta \mu v_{t-1})$. Theoretically optimal for smooth convex problems.

## Adaptive learning rates

The idea: **scale the learning rate per parameter**, based on the history of its gradients.

### AdaGrad (Duchi et al. 2011)
$$G_t = G_{t-1} + g_t^2, \qquad \phi_t = \phi_{t-1} - \frac{\eta}{\sqrt{G_t} + \varepsilon} g_t.$$
Parameters that have received large gradients get smaller updates. Works well for sparse problems; **decays too aggressively** for deep learning.

### RMSProp (Hinton, unpublished)
Exponential moving average instead of cumulative sum:
$$G_t = \beta G_{t-1} + (1-\beta) g_t^2, \qquad \phi_t = \phi_{t-1} - \frac{\eta}{\sqrt{G_t} + \varepsilon} g_t.$$
Solves AdaGrad's decay issue.

### Adam (Kingma & Ba 2015) — the de facto default
Combines momentum + RMSProp + bias correction:
$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad &\hat m_t &= \frac{m_t}{1-\beta_1^t}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad &\hat v_t &= \frac{v_t}{1-\beta_2^t}\\
\phi_t &= \phi_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
\end{aligned}$$

Defaults: $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}, \eta = 10^{-3}$.

**AdamW** (Loshchilov & Hutter 2019): decoupled weight decay (don't apply weight decay through the $\hat v$ scaling). De facto default for modern transformers.

## Learning rate schedules

Static $\eta$ rarely works for thousands of epochs. Common schedules:

- **Step decay:** $\eta \leftarrow \eta / 10$ every $k$ epochs.
- **Cosine annealing:** $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$.
- **Warmup + cosine:** linearly increase $\eta$ for the first $w$ steps, then cosine decay. Standard for transformers (helps stabilize training in the early high-curvature phase).
- **Cyclical LR (Smith 2017):** triangular waveform, can find super-convergence regimes.

## Second-order methods (and why we don't use them)

Newton: $\phi \leftarrow \phi - H^{-1} g$. Quadratically convergent for convex $f$. **Infeasible for deep nets** — $H$ has $D^2$ entries (terabytes for big models), and Hessian inversion is $O(D^3)$.

Practical compromises:
- **L-BFGS:** quasi-Newton, builds low-rank Hessian approximation. Works only for small/medium nets in full-batch.
- **K-FAC:** Kronecker-factored Hessian approximation. Used for second-order natural gradient in RL (e.g. TRPO uses similar ideas — see [[TRPO Surrogate Objective]]).
- **Shampoo / Sophia:** modern adaptive second-order methods becoming competitive with AdamW.

## Key intuitions to retain

- **Loss landscape is non-convex but practically tractable** due to gradient noise + over-parameterisation + good architectures (residual connections).
- **Saddle points dominate critical-point counts** in high dimensions but SGD escapes them naturally (any non-zero gradient noise eventually pushes off the saddle along a descent direction).
- **Adam is the default**, learning rate of $10^{-3}$ to $10^{-4}$, warmup + cosine schedule. Tweak only if it doesn't work.
- **Batch size and learning rate scale together** — larger batches need larger LR (linear scaling rule).
- **Why does it generalise?** Implicit regularisation from SGD noise + over-parameterisation. See [[Overparameterisation]].

## See also

- [[Initialisation]] — getting started in the right region of $\phi$-space.
- [[Regularisation]] — explicit techniques (L2, dropout, batch norm).
- [[Overparameterisation]] — the modern theory of why this works at all.
- [[Convexity]] (RL vault) — the structure that's *missing* from DL loss landscapes.
- [[OGD Regret Bound]] — what we lose without convexity (online setting).
