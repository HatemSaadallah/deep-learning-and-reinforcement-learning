# Performance and Overparameterisation

**Tags:** #dl #theory
**Source:** Lecture 7 (Tangherloni) — "Performance and overparameterisation"

Modern deep networks have **far more parameters than training examples** ($D \gg N$) and yet **generalise extremely well**. Understanding this requires examining bias, variance, the **double descent phenomenon**, and the role of **implicit regularisation**.

## Lecture outline

1. Noise, bias, and variance
2. Double descent
3. Overparameterised networks
4. Choosing hyperparameters

## Three causes of test error

An NN often performs (almost) perfectly on training data — this **does not mean it generalises**. Test errors have three distinct sources:

1. **Inherent uncertainty in the task** — irreducible noise in the data-generating process.
2. **Amount of training data** — too little data → high variance.
3. **Choice of model** (architecture + hyperparameters) — determines bias and capacity.

## MNIST-1D — the running example

A 1D version of MNIST used throughout the lecture:
- $x_i \in \mathbb{R}^{40}$ — horizontal offsets at 40 positions.
- 10 classes (digits 0–9).
- Templates randomly transformed + Gaussian noise → training set.

**Setup:** 2-hidden-layer MLP with $D = 100$ units, softmax output, multi-class cross-entropy, SGD with batch size 100, learning rate $0.1$, 6000 steps (150 epochs).

**Result:** training error → 0, but **test error stays high** — the model has not generalised.

## Bias-variance decomposition

In 1D regression, additive noise has variance $\sigma^2$:
$$y = \mu[x] + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2).$$

The **expected test loss** decomposes into three additive components:

| Source | Description |
|---|---|
| **Noise** ($\sigma^2$) | Inherent uncertainty in data generation. **Irreducible.** Multiple valid outputs for each input. |
| **Bias** | Systematic deviation from the mean of the function we are modelling — **due to model limitations** (not flexible enough). |
| **Variance** | Uncertainty in the fitted model **due to choice of training set**. Different training sets → slightly different results. |

$$\mathbb{E}[\text{test error}] \;=\; \sigma^2 + \text{bias}^2 + \text{variance}.$$

## The classical bias-variance trade-off

Classical story: increase model capacity →
- **Bias decreases** (more flexible model).
- **Variance increases** (more sensitive to training data).

Plot: test error as a U-shape — first decreases (bias dominates), then increases (variance dominates). Optimal capacity at the bottom of the U.

## Double descent

**Modern observation:** continue increasing capacity past the U-shape's minimum, and a **second descent** appears.

```
test error
   ▲
   │  ╱╲   ← classical U-shape
   │ ╱  ╲      peak at interpolation threshold
   │     ╲    ╱─────── second descent
   │      ╲╱
   └─────────────────────────────► capacity
   underparameterised  ↑  overparameterised
                       interpolation threshold
                       (training error = 0)
```

**Three regimes:**
1. **Classical / under-parameterised regime** — bias-variance trade-off, classical U.
2. **Critical regime** — at the **interpolation threshold**: the model has *just enough* capacity to perfectly fit (interpolate) the training data — training error is zero, **test error peaks**.
3. **Modern / over-parameterised regime** — past the interpolation threshold, **test error decreases again**, sometimes below the first minimum.

### Two phenomena combined

Double descent results from interaction of:
- **Bias-variance trade-off** (first descent + the peak).
- **Improvement past interpolation** (second descent) — test performance continues to improve along with capacity even after training error is perfect. **Why?**

## Inductive bias

Once the model has enough capacity to obtain training loss $\approx 0$, it fits the training data near-perfectly. **Further capacity cannot improve training fit.** Any change must occur **between training points**.

**Inductive bias** = the tendency of a model to **prioritise one solution over another** when extrapolating between data points.

**Inductive bias determines how NNs interpolate between data points.**

**Example: CNNs vs fully connected.** Same number of parameters; CNNs perform far better on images because their inductive bias is right for image data (translation equivariance, locality, weight sharing).

## Curse of dimensionality

The model's behaviour between data points is **critical** because in high-dimensional space, **training data is extremely sparse**.

**MNIST-1D example:** 40 dimensions, 10,000 samples. Quantize each dimension into 10 bins → $10^{40}$ bins total. With $10^5$ samples, there's only **1 point per $10^{35}$ bins**.

**Definition.** The tendency of high-dimensional space to overwhelm the number of data points is the **curse of dimensionality**. In high dimensions there are **small regions of data with significant gaps between them**.

## Why does double descent happen? — putative explanation

As we add capacity to the model, it **interpolates between the nearest data points increasingly smoothly**.

**At the interpolation threshold:** the model has *just enough* capacity to fit the training data exactly — but **must contort itself**, resulting in erratic predictions. This explains why the **peak is so pronounced**.

**Past the threshold:** extra capacity allows the model to interpolate smoothly even through sparse data — but **smooth interpolations generalise better**.

## Why smoothness? (the open question)

All curves that pass through the training data have zero training loss. **Why does the network prefer smooth ones?**

Three (uncertain) explanations:
1. **Network initialisation might encourage smoothness** — random-init functions tend to be smooth.
2. **The training algorithm "prefers" to converge to smooth functions** — implicit bias of SGD.
3. **The architecture's inductive bias** — e.g. CNNs implicitly favor translation-invariant functions.

> Any factor that biases a solution towards a subset of equivalent solutions is known as a **regulariser**. **The training algorithm acts as an implicit regulariser.**

See [[Regularisation]] for explicit vs implicit regularisation.

## Implicit regularisation in over-parameterised NNs

- Training algorithm can act as an implicit regulariser.
- **Extra model capacity describes areas with no training data** — these are where overparameterisation matters.
- **Regularisation favours smooth interpolation** between nearby points.
- Network initialisation may encourage smoothness.
- Training algorithm tends to converge to smooth functions.

## Weird properties of high-dimensional space

Two randomly sampled data points from a standard normal distribution are very **close to orthogonal** to one another (relative to the origin) **with high probability**.

This and related geometric facts mean that intuitions from 2D / 3D don't transfer — high-dim space is mostly empty, distances concentrate, random directions are nearly orthogonal.

## Theoretical foundations — Bubeck & Sellke 2021

*"A Universal Law of Robustness via Isoperimetry"*

**Smooth interpolation in $D$ dimensions requires $D$ times more parameters than mere interpolation.**

A trade-off exists between:
- **Number of parameters**
- **Lipschitz constant** (maximum rate of output change for small input changes)

**Implication:** current models for large datasets **may not be overparameterised enough**. Further increasing model capacity could be key to improving performance.

This formalises the "more parameters help" intuition behind the modern scaling-law era.

## Choosing hyperparameters

Deep NNs have **many** hyperparameters:
- Number of hidden layers, units per layer.
- Loss function.
- Optimisation algorithm + its parameters (learning rate, momentum, ...).
- Architectural choices.

**We don't have access to bias or variance** directly:
- Bias requires knowing the true function.
- Variance requires multiple independent datasets.

There's no way to tell a priori how much capacity is enough.

### The empirical approach (and the role of a validation set)

Hyperparameters are chosen empirically:
1. Train many models with different hyperparameters on the **training set**.
2. Measure their performance.
3. Select the best model.

**Critical: do not measure their performance on the test set!** Reserve the test set for final evaluation only.

**Solution: a third dataset — the validation set.**

```
[ Training set ] [ Validation set ] [ Test set ]
```

For every choice of hyperparameters:
- Train using the training set.
- Evaluate on the validation set.
- Select the model that worked best on validation.
- Final performance estimate on the test set (only **once**).

### Challenges

**Complex optimisation spaces:**
- Many hyperparameters are **discrete** (number of hidden layers).
- Conditional dependencies exist.
- **Cannot use gradient descent** for optimisation.
- Hyperparameter space is smaller than parameter space but **still too large for exhaustive search**.

**Techniques:**
- Intelligent sampling of hyperparameter space (Bayesian optimisation, random search, grid search).
- **Neural Architecture Search** (NAS) for structural optimisation, e.g. via neuroevolution.

## Summary

- Noise, bias, and variance — three additive sources of test error.
- Double descent — classical U + a second descent past the interpolation threshold.
- Inductive bias and the curse of dimensionality explain why over-parameterisation works.
- Bubeck-Sellke gives a theoretical lower bound: $D$× more parameters needed for smooth interpolation.
- Hyperparameter choice: training / validation / test split.

## See also

- [[Regularisation]] — explicit and implicit techniques (implicit regularisation is what makes overparameterisation work).
- [[Optimisation]] — SGD's implicit bias toward flat minima.
- [[Initialisation]] — init may encourage smoothness.
- [[Neural Network Fundamentals]] — bias-variance basics.
