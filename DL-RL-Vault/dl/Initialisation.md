# Initialisation

**Tags:** #dl #foundational
**Source:** Lecture 5 (Tangherloni) — "Initialisation strategies"

How to set the initial parameters $\phi_0$ of a neural network. Critical because poor initialisation causes **slow convergence**, **poor generalisation**, or **failed training** — particularly through **vanishing** or **exploding gradients** in deep networks.

## Lecture outline

1. Parameter initialisation — why it matters
2. Exploding vs vanishing gradients
3. He initialisation (mathematical derivation)
4. Comparison of initialisation techniques
5. Role of batch normalisation

## Standard building block — pre-activation analysis

Consider a layer with pre-activations
$$z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}.$$

Assume $W^{(\ell)}_{ij}$ are i.i.d. with $\mathbb{E}[W] = 0$ and $\mathrm{Var}(W) = \sigma_W^2$, and the previous activations $h^{(\ell-1)}_j$ are i.i.d. with $\mathbb{E}[h^{(\ell-1)}_j] = 0$ (we want this) and $\mathrm{Var}(h^{(\ell-1)}_j) = \sigma_h^2$.

Then
$$\mathrm{Var}(z^{(\ell)}_i) = n_{in}^{(\ell)} \cdot \sigma_W^2 \cdot \sigma_h^2$$
where $n_{in}^{(\ell)}$ is the layer's fan-in.

**Forward propagation of variance:**
$$\sigma_h^{(\ell)\,2} \;\propto\; n_{in}^{(\ell)} \cdot \sigma_W^{(\ell)\,2} \cdot \sigma_h^{(\ell-1)\,2}.$$

Iterating across $L$ layers, the variance is a **product**. If
- $n_{in} \sigma_W^2 > 1$: variance **explodes** exponentially in depth.
- $n_{in} \sigma_W^2 < 1$: variance **vanishes** exponentially in depth.

Backward pass: same story for the variance of $\partial L / \partial h^{(\ell)}$, but the relevant fan is $n_{out}$.

## Exploding gradients

- Occurs when weights are initialised with **high variance**.
- Gradients grow exponentially in deep networks.
- **Unstable training**: updates become too large, parameters diverge, NaNs appear.

## Vanishing gradients

- Occurs when weights are initialised with **small variance**.
- Gradients shrink exponentially as they propagate backward.
- **Learning becomes ineffective** — early layers stop updating, the deep stack effectively acts shallower.

## He (Kaiming) initialisation

Derived for **ReLU activations**. ReLU zeros out half the inputs (in expectation, under symmetric inputs), so only half the variance survives the non-linearity:
$$\mathrm{Var}(\mathrm{ReLU}(z)) = \tfrac{1}{2}\mathrm{Var}(z) \quad (\text{for symmetric, zero-mean } z).$$

To **preserve variance across layers**, the variance equation becomes
$$1 = \tfrac{1}{2}\, n_{in}\, \sigma_W^2 \;\Longrightarrow\; \boxed{\sigma_W^2 = \frac{2}{n_{in}}.}$$

**He-normal:** $W_{ij} \sim \mathcal{N}(0,\; 2/n_{in})$.
**He-uniform:** $W_{ij} \sim \mathcal{U}\!\bigl(-\sqrt{6/n_{in}},\, +\sqrt{6/n_{in}}\bigr)$.

This is the **default initialisation for any ReLU-based network** (CNNs, MLPs, the early layers of CV pipelines).

## Comparison of initialisation techniques

| Initialisation | Best for |
|---|---|
| **Zero init** | **Not recommended** — all neurons identical → no symmetry breaking → no learning. |
| **Random init** (e.g. $\mathcal{N}(0, 0.01)$) | Unstable for deep networks — wrong variance, exploding/vanishing. |
| **Xavier (Glorot)** | Sigmoid, tanh — preserves variance assuming symmetric, near-linear activations. $\sigma_W^2 = 2/(n_{in} + n_{out})$. |
| **He (Kaiming)** | ReLU, Leaky ReLU — accounts for the half-variance loss. $\sigma_W^2 = 2/n_{in}$. |
| **Orthogonal** | Stable but expensive — preserves variance and orthogonality through forward/backward passes; needs SVD or QR decomposition. |

**Orthogonal init** in detail:
- Weights drawn from an orthogonal matrix (often QR decomposition of a random Gaussian).
- Preserves orthogonality during forward and backward passes — variance of activations stays constant across layers exactly.
- Maintains stable gradients.
- **Expensive**: requires SVD or QR per layer at init time.

## More advanced initialisation strategies

### ConvolutionOrthogonal (Xiao et al. 2018)
Networks of up to **10,000 layers** can be trained without residual connections, using a carefully constructed orthogonal init for convolutional layers + tanh activations. *("Dynamical Isometry and a Mean Field Theory of CNNs")*.

### Fixup (Zhang et al. 2019)
Designed for **residual networks** without normalisation. Replaces BN's role purely via initialisation (scaling residual branches appropriately).

### T-Fixup (Huang et al. 2020)
Adapted to **transformers** — stabilises training and allows **removing LayerNorm**.

### DT-Fixup (Xu et al. 2021)
Allows transformers to be trained with **smaller datasets**.

## The role of batch normalisation

[[Regularisation|BatchNorm]] mitigates poor initialisation effects by **normalising activations within each layer**:
- In deep networks, BN reduces reliance on careful initialisation.
- Reduces "internal covariate shift" — the network can learn even with suboptimal init.

### BatchNorm and initialisation

- With BN, weight initialisation becomes **less critical**.
- Networks can train well even with simple initialisation.
- Proper initialisation **still improves training efficiency**, but isn't make-or-break.

## Bias initialisation

Almost always **zero**. Common exceptions:
- Output bias of classification: $\log(\pi_c)$ for class priors $\pi_c$ — speeds early training when classes are imbalanced.
- LSTM **forget gate** bias: set to 1 (encourages "remember by default" at init).

## Failure modes if you ignore initialisation

- **All weights zero** → all neurons compute the same thing → identical gradients → never breaks symmetry.
- **All weights identical** (non-zero) → same symmetry problem.
- **Variance too large** → exploding gradients, NaNs, training diverges.
- **Variance too small** → vanishing gradients, deep layers effectively learn nothing.

## Summary of the lecture

- Parameter initialisation
- Exploding vs vanishing gradients
- He initialisation (variance-preservation derivation for ReLU)
- Other initialisers (Xavier, Orthogonal, Fixup family)
- Role of batch normalisation

## See also

- [[Optimisation]] — what happens *from* the initial point.
- [[Regularisation]] — BatchNorm / LayerNorm reduce sensitivity to init.
- [[Transformer]] — transformers have their own init gotchas (T-Fixup).
- [[Neural Network Fundamentals]] — backprop and the chain rule that produces vanishing/exploding gradients.
