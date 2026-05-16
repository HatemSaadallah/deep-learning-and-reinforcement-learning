# Optimisation (Deep Learning)

**Tags:** #dl #optimisation #foundational
**Source:** Lecture 4 (Tangherloni) — "Fitting the models"

How to **fit** the parameters $\phi \in \mathbb{R}^D$ of a neural network — minimize the empirical loss
$$y_i = g[x_i] + \varepsilon_i, \qquad L[\phi] = \frac{1}{N}\sum_{i=1}^N \ell_i\bigl[f[x_i, \phi],\, y_i\bigr].$$

This is "learning the network's parameters", "training", or "fitting the model".

## General framework of optimisation

An **optimisation problem** is a triplet
$$\mathcal{P} = (S, c, \mathcal{N})$$
- $S$: search space of admissible configurations / candidate solutions.
- $c: S \to \mathbb{R}$: objective (aka cost or fitness) function.
- $\mathcal{N}: S \to 2^S$: neighbourhood structure (local connectivity).

The **fitness landscape** $\mathcal{L} = (S, \mathcal{N}, c)$ separates:
- **Topology** ($\mathcal{N}$): how points are connected.
- **Geometry / morphology** ($c$): shape of hills and valleys.

## Landscape vocabulary

- $x^*$ is a **local minimum** if $c(x^*) \leq c(x)$ for all $x \in \mathcal{N}(x^*)$.
- $x^*$ is the **global minimum** if $c(x^*) \leq c(x)$ for all $x \in S$.
- ⚠ Here $x$ denotes a candidate solution (the *parameters*), **not** the NN input.

**Topology can be studied via:**
- **Critical points** ($\nabla c = 0$): minima, maxima, **saddle points**.
- **Morse theory**: links the number of critical points and the topology of level sets.
- **Basins of attraction**: sets of points converging to the same local minimum under local descent.

## Convexity tests

- **1D:** $c$ is convex iff $c''(x) \geq 0$ everywhere.
- **General:** $c$ is convex iff the **Hessian** $H[\phi]$ is positive-semidefinite (all eigenvalues $\geq 0$) at every $\phi$.

NN loss landscapes are **non-convex** almost everywhere.

## Parameter vs function space

Define
$$\Phi: \Theta \to \mathcal{F}, \qquad \Phi(\phi) = f[\,\cdot\,, \phi].$$
- $\Theta = \mathbb{R}^D$: **parameter space**.
- $\mathcal{F} = \{f[\,\cdot\,, \phi] : \mathcal{X} \to \mathcal{Y} \mid \phi \in \Theta\}$: **function space**.

**Optimisation happens in parameter space, but generalisation is a property of the function space.**

**Different parameterisations can lead to the same function:**
$$f_{\theta_1} = f_{\theta_2} \;\nRightarrow\; \theta_1 = \theta_2.$$

This is why two networks with very different weights can compute identical functions, and conversely why two minima at the same loss can describe different functions.

## Landscape characteristics of NNs

1. **Highly non-convex** but with **connected regions of low loss**.
2. **Symmetries** (neuron permutations, scaling under ReLU homogeneity) create **degenerate minima** — entire manifolds of equivalent solutions.
3. **Overparameterisation tends to smooth the landscape** — see [[Overparameterisation]].

ResNet-56 illustrates dramatically: **with skip connections** the loss landscape is nearly convex around the minimum; **without** them, wildly bumpy (Li et al. 2018).

## Gradient flow → gradient descent

**Negative gradient flow (continuous):**
$$\dot\phi = -\nabla_\phi L.$$
This **decreases $L$ monotonically:**
$$\frac{\partial L}{\partial t} = \nabla_\phi L^\top \dot\phi = -\|\nabla_\phi L\|^2 \leq 0.$$

**Discretisation via Euler integration** with step $h$:
$$\phi^t = \phi^{t-1} - h\, \nabla_\phi L(\phi^{t-1}).$$
This is **gradient descent**.

### Challenge: discretisation

The discrete update can:
- **Converge** (small enough $h$),
- be **unstable** (oscillations, moderate $h$),
- **diverge** (large $h$).

For quadratic objectives the threshold is known. For NNs it's far more complex — the safe $h$ has to be found empirically.

## Sharp vs flat minima — and the flatness/generalisation debate

Convergence to **flat minima** has been linked to better generalisation in DL — but this is **debated**.

**Two definitions of flatness:**
1. **Gradient norms** — measures the steepness of the loss landscape near the minimum.
2. **Hessian eigenvalues** — smaller eigenvalues indicate a flatter minimum.

**Generalisation:** how well the model performs on unseen data.

### The Dinh et al. 2017 caveat ("Sharp Minima Can Generalize for Deep Nets")

NN **reparametrisation properties:** transformations of parameter values that **don't alter functional behaviour** (input-output mapping preserved). These reveal symmetries and invariances in parameter space.

**These reparametrisations can dramatically change sharpness without changing the function** — you can construct equivalent NNs with hugely varying sharpness, generalising equally well.

→ "Flat = good generalisation" is at best a parameter-space heuristic; the function-space view doesn't share it. Modern theory treats "flatness" as one of many proxies, not a definitive predictor.

## Stochastic Gradient Descent (SGD)

**Motivation:**
- Deal with large datasets.
- Avoid the slow training of full-batch GD.
- Early in training, **a full gradient update is unnecessary** — you can learn faster with smaller batches.

**Recipe.** Replace $\nabla L$ with an estimate from a **mini-batch** of size $B$:
$$\phi^t = \phi^{t-1} - h\, \nabla_\phi \hat L_B(\phi^{t-1}),\qquad \hat L_B(\phi) = \frac{1}{B}\sum_{i \in \text{batch}_t} \ell_i.$$

**Why SGD is the main workhorse of DL:**
- Improved generalisation performance.
- Avoids bad local minima and saddle points (the noise pushes off them).
- Empirically leads to **flatter minima**.

## Momentum

A modification to SGD that adds a momentum term — a **weighted combination of the current gradient and the previous step direction**:
$$\begin{aligned}
m^t &= \beta\, m^{t-1} + \nabla L(\phi^{t-1}) \\
\phi^t &= \phi^{t-1} - h\, m^t
\end{aligned}$$
- $\beta \in [0, 1)$ controls how much past direction matters (typically $0.9$).

Damps oscillations across high-curvature directions, accelerates along low-curvature ones.

### Nesterov accelerated momentum (NAG)

Evaluate the gradient at the **look-ahead point** rather than the current point:
$$m^t = \beta m^{t-1} + \nabla L(\phi^{t-1} - h\beta m^{t-1}).$$

Theoretically optimal first-order method for smooth convex problems.

## Adam — Adaptive Moment Estimation

Combines momentum + per-parameter adaptive learning rates + bias correction (Kingma & Ba 2015).

$$\begin{aligned}
m^t &= \beta_1 m^{t-1} + (1-\beta_1)\, g^t,\quad &\hat m^t &= \frac{m^t}{1-\beta_1^t}\\
v^t &= \beta_2 v^{t-1} + (1-\beta_2)\, (g^t)^2,\quad &\hat v^t &= \frac{v^t}{1-\beta_2^t}\\
\phi^t &= \phi^{t-1} - h\, \frac{\hat m^t}{\sqrt{\hat v^t} + \varepsilon}
\end{aligned}$$

Defaults: $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}, h = 10^{-3}$.

**AdamW** (Loshchilov & Hutter 2019): decoupled weight decay — the de facto default for modern transformers.

## Backpropagation

**Algorithm** for computing $\nabla_\phi L$ efficiently.

**Forward pass:** compute and **store** all intermediate activations $h^{(0)}, \dots, h^{(L)}$, the loss.

**Backward pass:** apply the chain rule layer-by-layer:
$$\frac{\partial L}{\partial h^{(\ell-1)}} = \Bigl(\frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}}\Bigr)^\top \frac{\partial L}{\partial h^{(\ell)}},\qquad \frac{\partial L}{\partial W^{(\ell)}} = \frac{\partial L}{\partial h^{(\ell)}} (h^{(\ell-1)})^\top.$$

### Pros and cons

| Pros | Cons |
|---|---|
| **Extremely efficient** — only matrix multiplies (the most computationally demanding step) and ReLU thresholding | **Memory-hungry** — all forward-pass intermediates must be stored, limiting model size |
| Easily parallelisable across batches | **Mainly sequential** along the chain — problematic if the model can't fit on one machine |

## Algorithmic differentiation (autograd)

Modern DL frameworks compute derivatives **automatically**:
- You specify the model and loss.
- Each component knows how to compute its own derivative.
- The framework tracks the network's operation sequence (the **computational graph**).
- It can then compute forward and backward passes via the chain rule.

**Works with branches** as long as the computational graph is acyclic. Generalises beyond NNs — autograd works on **any differentiable computation graph**.

## Computational optimisations

**Memory:**
- **Gradient checkpointing** — store only selected activations, recompute the rest during backward.
- **In-place operations** where possible.
- Batch-size tuning.

**Computation:**
- Parallelisation across batches.
- **Fused operations** (e.g. matmul + bias addition).
- **Mixed-precision training** (fp16/bf16).
- Optimise memory access patterns.

## PyTorch recipe

```python
# Random parameters
w = torch.randn(d, requires_grad=True)
# Fixed random input/target
x = torch.randn(N, d)
y = torch.randn(N)
# Optimiser
optimizer = torch.optim.SGD([w], lr=0.01)

# Training loop
for step in range(T):
    optimizer.zero_grad()
    y_hat = x @ w
    loss = ((y_hat - y) ** 2).mean()
    loss.backward()         # autograd computes ∇_w loss
    optimizer.step()        # SGD update
```

**Autograd works on any differentiable computation graph** — you don't need to build a neural network.

## Summary of the lecture

1. Optimisation in deep learning (landscape, framework).
2. **Sharp vs flat minima** — flatness and generalisation (debated, see Dinh 2017).
3. Gradient Descent and Stochastic Gradient Descent.
4. Momentum and Adam.
5. Backpropagation + algorithmic differentiation.

## See also

- [[Initialisation]] — getting started in the right region of $\phi$-space.
- [[Regularisation]] — explicit and implicit techniques.
- [[Overparameterisation]] — why this all works empirically.
- [[Neural Network Fundamentals]] — backprop derivation, MLE view of losses.
- [[Convexity]] (RL vault) — the structure that's missing from DL loss landscapes.
