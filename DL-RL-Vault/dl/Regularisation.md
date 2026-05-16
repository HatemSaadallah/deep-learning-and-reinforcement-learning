# Regularisation

**Tags:** #dl #foundational
**Source:** Lecture 6 (Tangherloni) — "Regularisation"

A **family of methods that reduce the generalisation gap** between training and test performance. Technically "regularisation" means adding terms to the loss function; in practice the term is used for **any strategy that improves generalisation**.

## The problem: overfitting

Significant performance gap between training and test data:
- **Overfitting:** the model captures statistical peculiarities of the training data not representative of the true mapping.
- Networks are unconstrained in areas with no training data → suboptimal predictions on unseen inputs.

## Lecture outline

1. **Explicit regularisation** — added loss terms or modifications
2. **Implicit regularisation** — biases of the training procedure
3. **Heuristics** for improving performance
4. **Transfer learning, self-supervised learning, data augmentation**
5. Comparisons and practical tips

## Explicit regularisation

### L2 regularisation (weight decay)

Add $\lambda \sum_j \phi_j^2$ to the loss:
$$\hat\phi = \arg\min_\phi \Bigl[L(\phi) + \lambda \sum_j \phi_j^2\Bigr].$$

Update rule becomes
$$\phi_{t+1} = (1 - 2\eta\lambda)\, \phi_t - \eta\, \nabla L(\phi_t).$$
The factor $(1 - 2\eta\lambda)$ **shrinks weights toward zero each step** — hence "weight decay".

Pros: simple, well-understood. Cons: may nullify performance if too strong.

### L1 regularisation

Add $\lambda \sum_j |\phi_j|$. Encourages **sparse weights** (many exactly zero). Less common than L2 in deep learning; used when sparsity matters (pruning, interpretability).

### Batch Normalisation as regularisation

**Idea:** "move" all minibatches to a standard location — mean zero and unit standard deviation. This eliminates **covariate shift** between batches.

**Operation:** for each activation $h$ and batch $\mathcal{B}$:
$$\hat h = \frac{h - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \varepsilon}}, \qquad y = \gamma\, \hat h + \delta$$
where:
- $\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}^2$: **empirical mean and variance** across the batch.
- $\varepsilon$: small constant preventing division by zero.
- $\gamma, \delta$: learned **scale and shift** parameters, fitted using all batches.

**Why BatchNorm works (multiple benefits):**
1. Makes the loss surface **smoother and slower-changing** → **higher learning rates** become safe (Santurkar et al. 2018 "How Does Batch Normalization Help Optimization?"). Higher LR → better test performance.
2. Acts as a **regulariser through noise injection** — batch statistics vary across iterations, providing implicit noise.
3. Decreases variation in loss and gradients.

**Key property:** the normalisation depends on the **batch statistics** — different normalisation per training iteration.

### BatchNorm vs LayerNorm

| | BatchNorm | LayerNorm |
|---|---|---|
| Normalises across | the **batch** dimension | the **feature** dimension |
| Per-example? | no (uses batch statistics) | yes (independent for each sample) |
| Small batch size? | unreliable | fine |
| Used in | CNNs | Transformers, RNNs |

### Early stopping

Stop training before convergence:
- Effectively reduces the **effective model complexity**.
- Simple to implement, no computational overhead.
- Requires a validation set.

### Dropout

Randomly set a subset of hidden units to **zero** at each SGD iteration.
- The network becomes **less dependent on any single hidden unit**.
- **Encourages weights to have smaller magnitudes** — the change in the function due to presence/absence of a hidden unit is reduced.
- Can be interpreted as **applying multiplicative Bernoulli noise** to activations.

### Applying noise more generally

Dropout is one instance of a broader idea — inject noise to encourage smoothness.

**Noise on activations** (dropout): smooths the learned function w.r.t. units.

**Noise on inputs:** smooths the learned function w.r.t. inputs — equivalent to data augmentation in many cases.

**Noise on weights:** the network must make sensible predictions even under small weight perturbations. Training converges to **flat minima** where small weight changes don't change performance.

**Noise on labels (label smoothing):** in multiclass prediction, MLE pushes the network to predict the correct class with absolute certainty — final activations (pre-softmax) are pushed to extreme values. **Perturb labels** (replace $y = e_c$ with a slight mixture) to discourage overconfidence.

## Implicit regularisation

**Neither GD nor SGD moves neutrally to the minimum** — each exhibits preferences for some solutions over others. This is **implicit regularisation**.

### GD as a discretisation of gradient flow

Continuous-time GD with infinitesimal steps: $\dot\phi = -\nabla L(\phi)$.

GD approximates this via discrete steps of size $\alpha$:
$$\phi_{t+1} = \phi_t - \alpha\, \nabla L(\phi_t).$$
The discretisation causes a **deviation from the continuous path**. The discrete dynamics implicitly minimise a slightly different objective that includes a gradient-norm penalty $\|\nabla L\|^2$ — a form of implicit regularisation.

### SGD implicit bias

Stochastic gradient noise biases SGD toward **flat minima** — see [[Optimisation]] and [[Overparameterisation]] for the connection to generalisation.

## Heuristics for improving performance

### Ensembling

A **group of models** combined to predict — outputs often averaged or voted.
- Different random initialisations → different models.
- **Bagging:** resample training data with replacement, train one model per resample, average.

Reduces variance, improves stability. Cost: train and run multiple models.

## Transfer learning, self-supervised learning, data augmentation

### Transfer learning

When training data is limited, exploit data from **related tasks**:
1. **Pre-train** the network on a related secondary task with much more data.
2. **Adapt** to the original task — replace the last layer(s) with new task-specific layers.
3. Either **freeze** the pretrained part (linear probe) or **fine-tune** the entire model.

The pretrained network functions as a feature extractor; the original task only needs to learn the mapping from those features.

### Self-supervised learning

Create large amounts of "free" labelled data via pretext tasks. Two families:

**Generative SSL** — mask part of the sample, predict the missing part. E.g. **Masked Language Modelling** (BERT, MAE).

**Contrastive SSL** — pairs of examples with commonalities compared to unrelated pairs. E.g. **Next Sentence Prediction**, SimCLR, CLIP.

### Data augmentation

Expand the dataset by transforming each sample **while keeping the label** the same.

Common augmentations:
- **Images:** flip, crop, color jitter, rotation, MixUp, CutMix, AutoAugment.
- **Audio:** time/frequency masking (SpecAugment).
- **Text:** synonym replacement, back-translation.

Equivalent to adding noise to inputs in a controlled way.

## Regularisation overview map

```
                Make the function smoother          Increase the data
                ───────────────────────             ─────────────────
                Apply noise to inputs               Data augmentation
                Apply noise to outputs (label sm.)  Transfer learning
   EXPLICIT     Apply noise to weights              Multi-task learning
                Early stopping
                Ensembling                                IMPLICIT
                Bayesian approach          ←  Dropout  →  regularisation
                                              (combines both views)
```

## Choosing regularisation techniques

| Scenario | Strategy |
|---|---|
| Small datasets | Transfer learning + extensive data augmentation + strong regularisation (L2, dropout) |
| Large datasets | Focus on implicit regularisation; use BatchNorm; lighter explicit regularisation |
| Complex architectures | BatchNorm for deep nets; combine multiple techniques; residual connections |
| Training issues | Start with BatchNorm for optimisation; add dropout for overfitting; early stopping as final safeguard |

## Comparison table

| Technique | Pros | Cons | Best use case |
|---|---|---|---|
| L2 regularisation | Simple, well-understood | May nullify performance if too strong | General purpose |
| BatchNorm | Higher LR; better optimisation | Memory-intensive; batch-size dependent | Deep networks |
| Dropout | Easy; no new parameters | Slower convergence | Large networks |
| Early stopping | Simple; no overhead | Needs validation set | Compute-limited |
| Transfer learning | Leverages external data; less training time | Needs similar-domain data | Limited target data |

## Recent research findings

**Model capacity:**
- Smooth interpolation requires more parameters.
- Test performance depends on **effective capacity**, not raw parameter count.
- **Double descent** phenomenon observed (see [[Overparameterisation]]).
- Different features learned at different speeds.

**Future directions:**
- Understanding implicit biases.
- Optimal capacity for specific problems.
- Architecture-specific regularisation.

## Summary

- Explicit regularisation: L1/L2, BatchNorm, dropout, early stopping, noise injection (inputs/weights/labels)
- Implicit regularisation: GD discretisation bias, SGD's preference for flat minima
- Heuristics: ensembling
- External data: transfer learning, self-supervised learning, data augmentation
- Choose by scenario (data size, architecture, training stability)

## See also

- [[Optimisation]] — SGD's implicit regularisation is a property of the optimiser.
- [[Initialisation]] — interacts with BatchNorm (BN reduces sensitivity to init).
- [[Overparameterisation]] — implicit regularisation and double descent.
- [[Unsupervised Learning]] — self-supervised pretraining in depth.
- [[Transformer]] — uses LayerNorm in every block.
