# Initialisation

**Tags:** #dl #foundational

How to set the initial weights $\phi_0$ of a neural network. Critical because deep nets exhibit **vanishing** and **exploding** gradients at random/poor initialisations — the network either can't learn or diverges immediately.

## The vanishing / exploding gradient problem

Consider a deep net with $L$ layers, activations $h_l = \sigma(W_l h_{l-1})$. The gradient signal propagated backward through $L$ layers is a product of $L$ Jacobian factors. If each factor's spectral norm is:
- **< 1**: gradients **vanish** exponentially; early layers don't learn.
- **> 1**: gradients **explode** exponentially; weights diverge.

Random initialisation with too-small or too-large variance triggers exactly this. The job of careful initialisation is to **preserve the variance** of activations and gradients across layers.

## Xavier / Glorot initialisation (2010)

Designed for **symmetric activations** (tanh, sigmoid).

For a linear layer $h_l = W_l h_{l-1}$ with $W_l \in \mathbb{R}^{n_{out} \times n_{in}}$:
- Sample $W_{ij} \sim \mathcal{N}\!\left(0,\; \dfrac{2}{n_{in} + n_{out}}\right)$ (Xavier-normal),
- or uniform on $\left[-\sqrt{6/(n_{in}+n_{out})},\, +\sqrt{6/(n_{in}+n_{out})}\right]$ (Xavier-uniform).

**Derivation idea:** assume input has unit variance, activation function is linear near 0. Then $\mathrm{Var}(h_l) = n_{in} \cdot \mathrm{Var}(W_l) \cdot \mathrm{Var}(h_{l-1})$. Setting $\mathrm{Var}(W) = 1/n_{in}$ preserves forward variance. Backward pass requires $\mathrm{Var}(W) = 1/n_{out}$. Compromise: average gives $2/(n_{in} + n_{out})$.

## He / Kaiming initialisation (2015)

Designed for **ReLU** activations. Since ReLU zeros out half the inputs in expectation, only half the variance survives. Compensate:
$$W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{in}}\right).$$

This is the **default for any ReLU-based network** (CNNs, MLPs, etc.). Variants exist for Leaky ReLU, etc.

## Bias initialisation

Almost always **zero**. Exception: bias in the output layer of classification can be set to $\log(\pi_c)$ for class priors $\pi_c$ — speeds early training. Bias in the **forget gate** of LSTMs is set to 1 (encourages "remember by default" behavior at init).

## Initialisation for specific architectures

| Architecture | Recommended | Reason |
|---|---|---|
| MLP with ReLU | He-normal | matches ReLU non-linearity |
| MLP with tanh / sigmoid | Xavier-normal | matches symmetric non-linearity |
| CNN | He-normal | usually ReLU-based |
| Residual blocks (ResNet) | He + scale-by-$1/\sqrt{L}$ | keeps norm stable through $L$ residual blocks |
| Transformer | T-Fixup / Xavier with $1/\sqrt{L}$ scaling | self-attention destabilizes deep stacks otherwise |
| Embedding layers | $\mathcal{N}(0, 0.02)$ or scaled normal | scale tied to model dim $d_{\text{model}}$ |
| LSTM | Xavier for input gates, orthogonal for recurrent | preserves the recurrent dynamics |

## Orthogonal initialisation

Sample $W$ from an orthogonal matrix (via QR decomposition of a random Gaussian matrix). Preserves $\ell_2$ norm of forward and backward signals **exactly** when activations are linear. Good for very deep or RNN nets.

## Layer normalisation as an initialisation crutch

Adding **LayerNorm** ([[Regularisation|see Regularisation]]) or **BatchNorm** after each layer makes the network much less sensitive to initialisation. With normalisation layers, a wider range of initial weight scales works fine. Why standard transformers can use almost any reasonable init.

## Initialising the last layer

Output layer behavior depends on the task:
- **Regression**: linear, zero-init bias is fine.
- **Binary classification**: logit at zero gives $\sigma(0) = 0.5$ — symmetric init is fine. Bias to class log-prior for imbalanced data.
- **Softmax classification**: zero-init logits give uniform predictions — slow start unless temperatures are tuned.

## Failure modes if you ignore this

- **All weights zero** → all neurons compute the same thing → identical gradients → never breaks symmetry. **Never use zero init for weights.**
- **All weights identical (non-zero)** → same symmetry problem.
- **Variance too large** → activations saturate (sigmoid → ±1), gradients vanish from saturated units.
- **Variance too small** → effective network depth shrinks; deep layers compute zero.

## What "warmup" learning rate is doing

Warmup gradually increases the learning rate over the first $w$ steps. This compensates for the fact that early in training, gradients are unreliable (the network hasn't found its "preferred direction" yet) and large LR can wreck the initialisation.

For transformers, warmup interacts with initialisation: small init + warmup is the standard recipe to keep early-training dynamics stable.

## See also

- [[Optimisation]] — how the network moves *from* the initial point.
- [[Regularisation]] — BatchNorm/LayerNorm reduce sensitivity to init.
- [[Transformer]] — has its own initialisation gotchas.
