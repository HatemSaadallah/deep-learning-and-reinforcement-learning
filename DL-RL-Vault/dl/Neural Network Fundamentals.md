# Neural Network Fundamentals

**Tags:** #dl #foundational

The core mechanics every deep learning system rests on. This note collects the prerequisites referenced by [[Optimisation]], [[Regularisation]], and the rest of the DL vault: linear regression, shallow / deep nets, backpropagation, common loss functions.

## Linear regression as a starting point

**Model:** $\hat y = \phi_0 + \phi_1 x$ (one-dim) or $\hat y = \langle \phi, x\rangle + b$ (general).

**Loss:** sum of squared errors
$$L(\phi) = \sum_i (\hat y_i - y_i)^2.$$

**Closed-form solution** (normal equations): $\hat\phi = (X^\top X)^{-1} X^\top y$.

Linear regression is the **degenerate case** of a neural network with no hidden layers and identity activation. Everything else generalises this.

## Shallow neural network (one hidden layer)

$$h = \sigma(W^{(1)} x + b^{(1)}), \qquad \hat y = W^{(2)} h + b^{(2)}.$$

- $\sigma$ is a non-linear activation (ReLU, tanh, sigmoid).
- Without $\sigma$, the network collapses to a linear function.

**Universal Approximation Theorem (Cybenko 1989; Hornik 1991):** a shallow network with **one hidden layer** and **enough units** can approximate any continuous function on a compact set to arbitrary precision. (Existence claim; nothing about how to *find* the right parameters or about the size needed.)

## Deep neural network (DNN / MLP)

Stack $L$ layers:
$$h^{(\ell)} = \sigma(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}),\quad \ell = 1, \dots, L.$$

**Why deep > shallow:** even though shallow networks are universal approximators, deep networks approximate **many natural functions exponentially more efficiently** (Telgarsky, Eldan & Shamir). E.g., a function representable by a depth-$L$ ReLU net needs exponentially more units for a depth-$(L-1)$ approximation.

## Activation function essentials

For full table see [[Mixed DL Techniques]]. The basics:

- **ReLU** ($\max(0, x)$): standard for hidden layers since 2012.
- **Sigmoid** ($\sigma(x) = 1/(1+e^{-x})$): output of binary classifiers; **avoid** in hidden layers (vanishing gradients).
- **Tanh**: zero-centered, used in RNN gates.
- **Softmax** (vector-valued): output of multi-class classifiers, produces a probability distribution.

### ReLU homogeneity

ReLU is **positively-homogeneous of degree 1**: $\mathrm{ReLU}(\alpha z) = \alpha \mathrm{ReLU}(z)$ for $\alpha \geq 0$.

**Proof.** Case 1: $z \geq 0$. Then $\alpha z \geq 0$ (since $\alpha \geq 0$), so $\mathrm{ReLU}(\alpha z) = \alpha z = \alpha \cdot z = \alpha \cdot \mathrm{ReLU}(z)$. Case 2: $z < 0$. Then $\alpha z \leq 0$, so $\mathrm{ReLU}(\alpha z) = 0 = \alpha \cdot 0 = \alpha \cdot \mathrm{ReLU}(z)$. (Property fails for $\alpha < 0$.)

This homogeneity is what causes "weight scaling symmetries" in ReLU nets — scaling one layer up by $c$ and the next layer down by $1/c$ doesn't change the function. Has implications for regularisation and the loss landscape.

## Loss functions

| Task | Loss |
|---|---|
| Regression | $\sum_i (\hat y_i - y_i)^2$ (MSE) — assumes Gaussian noise model |
| Binary classification | $-\sum_i y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)$ (BCE) |
| Multi-class classification | $-\sum_i \sum_c y_{i,c} \log \hat p_{i,c}$ (CCE) — assumes categorical likelihood |

### Maximum likelihood interpretation

For Gaussian noise $y = g(x) + \varepsilon, \varepsilon \sim \mathcal{N}(0, \sigma^2)$:
$$\log p(y \mid x, \theta) = -\frac{(y - g_\theta(x))^2}{2\sigma^2} + \text{const}.$$
**Maximising the log-likelihood = minimising MSE.** Hence "the MLE of the parameters" coincides with "the parameters minimising MSE" in this setting.

Same logic for categorical / Bernoulli targets → cross-entropy. **"Loss = negative log-likelihood"** is the unifying view.

## Backpropagation

Algorithm for efficiently computing $\nabla_\phi L$.

**Forward pass:**
1. Compute and store all intermediate activations $h^{(0)}, h^{(1)}, \dots, h^{(L)}$.
2. Compute output $\hat y$ and loss $L$.

**Backward pass:** chain rule, propagated layer by layer.
For layer $\ell$:
$$\frac{\partial L}{\partial h^{(\ell-1)}} = \Bigl(\frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}}\Bigr)^\top \frac{\partial L}{\partial h^{(\ell)}},\qquad \frac{\partial L}{\partial W^{(\ell)}} = \frac{\partial L}{\partial h^{(\ell)}}\,(h^{(\ell-1)})^\top.$$

**Cost:** roughly **2× the forward pass** — for each layer, one matrix-multiply to compute $\partial L / \partial h^{(\ell-1)}$ and one for $\partial L / \partial W^{(\ell)}$.

**Why "stores intermediate values":** the backward pass uses the forward-pass activations, so they must be cached. This is what dominates memory cost during training.

## L2 regularisation, in detail

Add penalty $\lambda \sum_j \phi_j^2$ to the loss:
$$\hat\phi = \arg\min_\phi \Bigl[\sum_i \ell_i(\phi) + \lambda \sum_j \phi_j^2\Bigr].$$

**Gradient with regularisation:**
$$\nabla_\phi \mathcal{L}_{\text{reg}}(\phi) = \nabla_\phi \mathcal{L}_{\text{orig}}(\phi) + 2\lambda \phi.$$

**Update with SGD:** subtract this from $\phi$:
$$\phi_{t+1} = \phi_t - \eta\, \nabla_\phi \mathcal{L}_{\text{orig}}(\phi_t) - 2\eta\lambda\, \phi_t = (1 - 2\eta\lambda)\phi_t - \eta\nabla_\phi \mathcal{L}_{\text{orig}}.$$

The $(1 - 2\eta\lambda)$ factor "decays" the weights toward zero each step — hence "weight decay". Equivalent to L2 regularisation for standard SGD; **inequivalent** for adaptive optimisers (Adam) where you should use **AdamW** (decoupled weight decay).

## Universal Approximation Theorem — careful version

**Statement:** Let $\sigma$ be a non-constant, bounded, monotonically-increasing continuous function. Then for any continuous $f: K \to \mathbb{R}$ on a compact set $K \subset \mathbb{R}^n$ and any $\varepsilon > 0$, there exists a one-hidden-layer network
$$\hat f(x) = \sum_{i=1}^N \alpha_i\, \sigma(\langle w_i, x\rangle + b_i)$$
with $\sup_{x \in K} |f(x) - \hat f(x)| < \varepsilon$.

**Caveats:**
- Existence claim, not constructive.
- $N$ may need to be **astronomically large**.
- Says nothing about generalisation, only training-set approximation.
- Original proof required specific $\sigma$ (sigmoid). Extended to most non-polynomial activations including ReLU.

**Why deep > shallow** is **not** an UAT statement — it's an *efficiency* statement (depth-$L$ functions cheap for depth-$L$ nets, expensive for shallower).

## See also

- [[Optimisation]] — how to actually minimise the loss.
- [[Initialisation]] — getting started.
- [[Regularisation]] — controlling generalisation.
- [[Mixed DL Techniques]] — practical recipes.
