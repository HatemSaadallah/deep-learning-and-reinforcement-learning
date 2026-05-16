# GANs — Generative Adversarial Networks

**Tags:** #dl #generative #architecture

Generative models trained by playing a **two-player adversarial game** (Goodfellow et al. 2014):
- A **generator** $G$ tries to produce realistic samples from a latent noise $z$.
- A **discriminator** $D$ tries to distinguish real data from $G$'s output.

At equilibrium, $G$'s distribution matches the data distribution. GANs were SOTA for image generation 2014–2020 before being eclipsed by [[Diffusion Models]].

## The minimax game

Original objective (Goodfellow 2014):
$$\min_G \max_D \;\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].$$

- Inner max: $D$ tries to assign $1$ to real, $0$ to fake.
- Outer min: $G$ tries to fool $D$ (make $D(G(z))$ close to $1$).

At the optimum (Goodfellow's analysis), $G(z) \sim p_{\text{data}}$ and $D^* \equiv 1/2$ everywhere. **In practice, training is notoriously unstable** — see below.

## Training dynamics

Alternating SGD:
```
for each step:
    sample real batch x_real
    sample noise z, generate x_fake = G(z)
    update D to maximize log D(x_real) + log(1 - D(x_fake))
    update G to minimize log(1 - D(G(z)))  [or maximize log D(G(z)), see below]
```

**Non-saturating loss for G:** the gradient of $\log(1 - D(G(z)))$ vanishes when $D$ is confident → $G$ doesn't learn. Goodfellow recommended maximizing $\log D(G(z))$ instead — stronger gradients early in training.

## Failure modes of GAN training

1. **Mode collapse.** $G$ outputs only a few modes from the data distribution (e.g. only 2 of 10 MNIST digits). $D$ chases, $G$ shifts mode, both oscillate.
2. **Vanishing gradients.** If $D$ wins too fast, $G$ has no signal.
3. **Non-convergence.** Min-max optimisation on non-convex landscapes — equilibria need not be reached. Often produces oscillations.
4. **Sensitivity to hyperparameters.** A learning-rate tweak can crash training entirely.

## Key variants

### DCGAN (Radford et al. 2015)
Convolutional architecture guidelines: strided convs (no pooling), BatchNorm, ReLU in G / LeakyReLU in D. First GANs that reliably produced recognizable images.

### WGAN — Wasserstein GAN (Arjovsky et al. 2017)
Use the **Wasserstein-1 distance** (earth mover's distance) instead of JS divergence:
$$\min_G \max_{D \in 1\text{-Lipschitz}} \mathbb{E}[D(x)] - \mathbb{E}[D(G(z))].$$

- $D$ becomes a "critic" (no sigmoid, outputs real numbers).
- Constrained to be 1-Lipschitz (originally via weight clipping, **WGAN-GP** does it via gradient penalty).
- Much more stable training, meaningful loss curves.

### Conditional GAN (cGAN, Mirza & Osindero 2014)
Condition $G$ and $D$ on a label $y$: $G(z, y)$, $D(x, y)$. Enables class-conditional generation.

### StyleGAN (Karras et al. 2019–2021)
- Generator architecture with **adaptive instance normalisation** modulated by a learned "style vector".
- Progressive growing during training.
- StyleGAN2/3: SOTA face generation; FID scores on FFHQ that took years to beat.

### CycleGAN (Zhu et al. 2017)
Unpaired image-to-image translation: photo ↔ painting, horse ↔ zebra. Uses two generators ($A \to B$ and $B \to A$) with a **cycle-consistency loss** $\|G_{BA}(G_{AB}(x)) - x\|$.

### BigGAN (Brock et al. 2018)
Scale GANs to ImageNet at high resolution. Class-conditional. Showed that with enough scale + tricks (orthogonal regularisation, truncation), GANs can produce diverse, high-quality images at scale.

## GAN evaluation

Tricky because there's no likelihood. Standard metrics:
- **Inception Score (IS)**: $\exp(\mathbb{E}_x[D_{\text{KL}}(p(y|x) \| p(y))])$. Higher = more diverse and class-conditional confidence. Limited (only sees classifier).
- **Fréchet Inception Distance (FID)**: distance between Gaussian fits of Inception features for real vs. generated. Lower = better. Modern default.
- **Precision / Recall** (Sajjadi et al. 2018): separately measure sample quality (precision) and diversity coverage (recall).

## Why GANs lost the throne

By 2021, [[Diffusion Models]] surpassed GANs in quality and beat them on every fairness benchmark. Reasons:
- Diffusion has stable training (just MSE).
- Diffusion has a **likelihood** — better theoretical grounding.
- Diffusion scales better with compute.

GANs remain useful for:
- **Fast inference** (one forward pass through $G$ vs. dozens of denoising steps).
- **Specific applications** like superresolution (ESRGAN), face generation (StyleGAN).
- **Conditional generation** in low-data regimes.

## See also

- [[VAEs]] — alternative generative model with explicit likelihood.
- [[Diffusion Models]] — current SOTA, has displaced GANs for most image gen.
- [[Optimisation]] — GAN training relies on adaptive optimisers and careful LR balancing.
- [[Regret and Equilibria]] (RL vault) — connection: GAN training is a zero-sum game; no-regret dynamics theoretically converge to Nash.
