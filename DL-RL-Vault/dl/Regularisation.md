# Regularisation

**Tags:** #dl #foundational

Techniques to **reduce the gap between training and test loss** — i.e. to prevent overfitting. In deep learning the bias/variance trade-off is mediated less by model size than by training regime, so "regularisation" is a broad collection of tricks affecting many parts of the pipeline.

## Two types: explicit vs. implicit

| | Examples |
|---|---|
| **Explicit regularisation** — an added loss term or constraint | L2 / L1 weight decay, dropout, label smoothing, mixup, cutmix |
| **Implicit regularisation** — properties of the training procedure that bias toward simpler / flatter solutions | SGD noise, early stopping, batch norm, large learning rates, architecture choices |

The deep learning community has gradually realized that **implicit regularisation often matters more than explicit** — see [[Overparameterisation]].

## Explicit regularisation

### L2 (Tikhonov, "weight decay")

Add $\lambda \sum_j \phi_j^2$ to the loss. Equivalently, update rule becomes
$$\phi_{t+1} = (1 - 2\eta\lambda)\, \phi_t - \eta \nabla L(\phi_t).$$
The factor $(1 - 2\eta\lambda)$ **shrinks weights toward zero each step** — "weight decay". In Adam this should be **decoupled** (AdamW, Loshchilov 2019) rather than added through the adaptive denominator.

Typical $\lambda$: $10^{-4}$ to $10^{-2}$.

### L1 (Lasso)

Add $\lambda \sum_j |\phi_j|$. Encourages **sparse weights** (many exactly zero). Less common in deep learning than in linear models. Used for sparsity-targeted applications (pruning, interpretability).

### Dropout (Srivastava et al. 2014)

During training, **mask each unit independently with probability $p$** (typically $p \in [0.1, 0.5]$):
$$h_i \leftarrow \frac{1}{1-p} \cdot b_i \cdot h_i, \quad b_i \sim \mathrm{Bernoulli}(1-p).$$
At test time, all units active (no scaling needed because of the $1/(1-p)$ at train).

**Interpretation:** approximates a Bayesian ensemble over exponentially many "thinned" sub-networks (Gal & Ghahramani 2016).

**Variants:** spatial dropout (for CNNs, drop whole channels), DropConnect (drop weights instead of activations), DropPath / stochastic depth (drop residual branches).

### Label smoothing

Replace one-hot label $y = e_c$ with $y_c = 1-\alpha$, $y_{c' \neq c} = \alpha/(K-1)$ where $K$ is the number of classes and $\alpha \in [0.05, 0.1]$. Prevents the network from becoming overconfident.

### Data augmentation

| Domain | Augmentations |
|---|---|
| Images | flip, crop, color jitter, rotation, MixUp, CutMix, AutoAugment |
| Audio | time/frequency masking (SpecAugment), pitch shift |
| Text | synonym replacement, back-translation, random masking (used in BERT pretraining) |

**Mixup** (Zhang et al. 2018): train on convex combinations $\tilde x = \lambda x_i + (1-\lambda) x_j$, $\tilde y = \lambda y_i + (1-\lambda) y_j$. Strong regulariser, "linearises" model behavior between training examples.

## Implicit regularisation

### Early stopping

Stop training when validation loss starts increasing, even if training loss is still decreasing. Cheap and surprisingly effective. Equivalent to L2 regularisation in some convex cases (Friedman & Popescu).

### Batch Normalisation (BN, Ioffe & Szegedy 2015)

After a linear layer, normalise the **batch statistics** of the activations:
$$\hat h_i = \frac{h_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}},\qquad y_i = \gamma \hat h_i + \beta$$
where $\mu_B, \sigma_B^2$ are computed across the mini-batch, and $\gamma, \beta$ are learnable parameters per channel.

**Effects:**
- Reduces "internal covariate shift" (the original motivation, debated)
- Acts as a regulariser (each example sees a slightly different effective network due to batch noise)
- Smoothens the loss landscape (Santurkar et al. 2018) — allows larger learning rates

**Test time:** use running averages of $\mu, \sigma$ from training.

**Failure mode:** small batch sizes (< 16) make $\mu_B, \sigma_B$ unreliable. Replaced by:

### Layer Normalisation (LN, Ba et al. 2016)

Normalise across **features** rather than across batches. Default choice in transformers, RNNs, and any setting where batch size is small/variable.

$$\hat h_i = \frac{h_i - \mu_f}{\sqrt{\sigma_f^2 + \varepsilon}}$$
where $\mu_f, \sigma_f$ are computed across the feature dimension of a single example.

Modern transformers (e.g. LLaMA) use **RMSNorm** — even simpler, drops the mean centering.

### Other normalisations

- **GroupNorm** — normalise across feature groups, used in detection (small batches).
- **InstanceNorm** — per-example, per-channel; used in style transfer.
- **WeightNorm** — normalise the *weights* directly.

### Stochastic Gradient noise

The noise of mini-batch SGD acts as implicit regularisation, biasing toward **flat minima**. Flat minima are conjectured to generalise better — "sharp minima are sensitive to perturbations of the weights, flat ones aren't." See [[Overparameterisation]].

## Architectural regularisation

- **Residual / skip connections** — make optimisation easier, smooth landscape, allow much deeper nets.
- **Attention heads** — limit per-head complexity by splitting features into smaller subspaces.
- **Convolution** — translational symmetry baked into the architecture, equivalent to massive weight sharing → fewer effective parameters.
- **Pooling layers** — provide spatial invariance.

## What actually matters in modern practice

| Technique | Used? |
|---|---|
| L2 / weight decay | ✅ usually $10^{-4}$ with AdamW |
| Dropout | ⚠️ less common in transformer training; standard in MLP heads |
| BatchNorm | ✅ in CNNs |
| LayerNorm | ✅ in transformers (every layer) |
| Data augmentation | ✅ essential in CV; emerging in NLP |
| Label smoothing | ✅ for classification, $\alpha = 0.1$ |
| Early stopping | ✅ universal |
| Large batch SGD with linear LR scaling | ✅ for distributed training |

## See also

- [[Optimisation]] — interacts strongly with regularisation (noise from SGD is itself a regulariser).
- [[Initialisation]] — BatchNorm reduces sensitivity to init.
- [[Overparameterisation]] — why implicit regularisation works.
- [[Transformer]] — LayerNorm everywhere.
