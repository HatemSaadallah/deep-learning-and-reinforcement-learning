# Overparameterisation

**Tags:** #dl #theory

Modern deep nets have **far more parameters than training examples** ($D \gg N$). Classical statistical learning theory predicts this should **overfit catastrophically**. In practice these networks **generalise extremely well**. The mismatch motivated a new theoretical framework: the **overparameterisation regime**.

## Classical bias-variance vs. modern double descent

The classical curve: as model capacity increases, training loss decreases monotonically, test loss first decreases (lower bias) then increases (higher variance) — the famous **U-shape**.

The modern phenomenon: as model capacity continues to increase past the **interpolation threshold** (the point where training loss hits zero), test loss **decreases again** — sometimes below its first minimum!

```
Test loss
   ▲
   │
   │     U-shape          \    SECOND DESCENT
   │     (classical)       \   (modern overparameterised)
   │  ___                   \_______________
   │ /   \  (peak)            (interpolation
   │/     \                    threshold)
   └────────┴────────────────────────────────────► capacity
            ↑ N
```

**Double descent** (Belkin et al. 2019; Nakkiran et al. 2020) explains why modern practice — train extremely large models — works.

## Why does overfitting fail to happen?

The "deep learning generalisation puzzle". Several non-mutually-exclusive explanations:

### 1. Implicit regularisation of SGD

Among all global minimisers (which exist in abundance for overparameterised models), SGD prefers **flat ones**:
- Flat minima have wide basins, are robust to weight perturbations.
- Flat minima correspond to **simple functions** in the bias-variance sense.
- Sharp minima generalise poorly (Keskar et al. 2017).

### 2. Neural Tangent Kernel (NTK)

In the infinite-width limit, a NN trained with gradient descent behaves like a **kernel regression** with a specific kernel (the NTK, Jacot et al. 2018) — and kernel regression has well-understood generalisation theory. The NTK regime explains *some* behaviors but **does not capture feature learning**.

### 3. Lottery Ticket Hypothesis (Frankle & Carbin 2019)

A randomly-initialised dense network contains **sparse subnetworks** ("winning tickets") that, when trained in isolation, reach comparable accuracy. The overparameterised dense network is essentially a "search container" for finding good sparse functions.

### 4. Architectural inductive biases

- **CNNs** are biased toward translation-equivariant functions.
- **Transformers** are biased toward "long-range pattern matching" through attention.
- **Residual connections** bias toward functions close to identity.

These priors restrict the *effective* hypothesis class far below the parameter count.

## The interpolation threshold

The point where training loss = 0. Past this, the **training loss can't decrease further** but the **model can still change** (gradient descent moves through the zero-loss manifold). The choice of which zero-loss solution to settle on is what determines test performance.

For classification, this corresponds to the minimum-margin classifier (Soudry et al. 2018). For regression, the minimum-norm interpolant. These have known generalisation properties.

## Practical implications

1. **Larger models often generalise better** — explicit in scaling laws (Kaplan et al. 2020; Hoffmann et al. 2022 / Chinchilla).
2. **Don't fear training to zero training loss** — common pre-modern advice to "stop before overfitting" is mostly wrong for big models.
3. **Architecture and optimiser choices matter more than parameter count** — same $D$ with different inductive biases yields very different generalisation.

## Scaling laws

For LLMs (Kaplan et al. 2020):
$$L(N, D) \approx L_\infty + \frac{a}{N^{\alpha_N}} + \frac{b}{D^{\alpha_D}}$$
where $N$ = model params, $D$ = dataset tokens. Empirical exponents $\alpha_N \approx 0.34, \alpha_D \approx 0.28$.

**Chinchilla (Hoffmann et al. 2022)** revised: should scale $N$ and $D$ in roughly equal proportion. For a fixed compute budget $C \approx 6 N D$, the optimal allocation has $N \propto C^{0.5}$ and $D \propto C^{0.5}$ (rather than the earlier $N$-heavy split).

These scaling laws underpin the entire LLM-scaling era of deep learning.

## Saddle points are *not* the obstacle

In low dim, NN training looked stuck at saddle points (random matrix arguments). In practice:
- SGD noise pushes off saddles easily.
- Many "saddles" are actually **strict saddles** (have at least one direction with strictly negative curvature) — easily escapable.

## Open questions

- **Why does NTK regime fail to capture feature learning?** ML researchers' current best theory of training dynamics still misses what makes deep nets actually useful.
- **Can we predict generalisation from training dynamics?** Tools like sharpness, NTK trajectory, "training as compression" are partial answers.
- **What's the right notion of "complexity" for an over-parameterised model?** Parameter count is the wrong proxy. Norm-based bounds (Bartlett, Foster, Telgarsky) are tighter but still loose.

## See also

- [[Optimisation]] — SGD's implicit bias is at the heart of this.
- [[Regularisation]] — explicit techniques that complement implicit regularisation.
- [[Initialisation]] — sets the starting point that SGD bias acts from.
