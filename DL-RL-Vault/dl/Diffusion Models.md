# Diffusion Models

**Tags:** #dl #generative #architecture

Generative models that learn to **reverse a Gaussian-noise corruption process**. Current SOTA for image, video, audio, and 3D generation. Used in DALL-E 3, Stable Diffusion, Midjourney, Sora, AlphaFold 3, etc.

## The two processes

### Forward (noising) — fixed, no parameters

Given clean data $x_0$, gradually add Gaussian noise over $T$ steps:
$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\bigl(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I\bigr).$$

The variance schedule $\beta_t$ is small (e.g. $10^{-4}$ to $0.02$) so each step adds a little noise. After $T$ steps ($T \approx 1000$), $x_T \approx \mathcal{N}(0, I)$.

**Marginal:** for any $t$, can sample directly without intermediates:
$$q(x_t \mid x_0) = \mathcal{N}\bigl(\sqrt{\bar\alpha_t}\, x_0,\; (1-\bar\alpha_t) I\bigr),$$
where $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s\leq t}\alpha_s$. This is what lets us train without simulating the full chain.

### Reverse (denoising) — learned

A neural net $\epsilon_\theta(x_t, t)$ tries to **predict the noise** that was added at step $t$. Equivalently, predict the clean image $x_0$ given a noised version $x_t$ at known $t$.

## DDPM training (Ho et al. 2020)

Simple, stable loss:
1. Sample $x_0$ from data.
2. Sample $t \sim \mathrm{Uniform}(\{1, \dots, T\})$.
3. Sample $\varepsilon \sim \mathcal{N}(0, I)$.
4. Form $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon$.
5. Compute loss
$$\mathcal{L} = \|\varepsilon - \epsilon_\theta(x_t, t)\|^2.$$

**Just MSE between true noise and predicted noise.** No adversarial games, no KL divergence terms in the loss, no log-determinants. Massively more stable than [[GANs|GAN]] training.

## Sampling (generation)

Start from $x_T \sim \mathcal{N}(0, I)$. For $t = T, T-1, \dots, 1$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t)\Bigr) + \sigma_t z,\quad z \sim \mathcal{N}(0, I).$$

After $T$ steps, $x_0$ is a generated sample. **$T \sim 1000$ steps is slow** (e.g. 30 seconds per image on a GPU) — hence the proliferation of fast samplers.

## Fast samplers

The vanilla DDPM sampler has been replaced by:

- **DDIM** (Song et al. 2021) — deterministic, can skip steps, 10–50 steps for comparable quality.
- **DPM-Solver** (Lu et al. 2022) — high-order numerical solver for the underlying ODE. 10–20 steps.
- **Consistency models** (Song et al. 2023) — distill to a **single-step** generator. Trades quality for speed.
- **Flow matching** (Lipman et al. 2023) — unified framework, similar performance, simpler theory.

## Score-based view (Song et al. 2019, 2021)

A continuous-time perspective: define a forward SDE $dx = f(x, t)dt + g(t) dW$. The reverse process is also an SDE involving the **score function** $\nabla_x \log p_t(x)$. The neural net learns this score.

This unifies DDPM with **score-based generative modeling** — they're the same thing in continuous time. The framework also yields:
- **Probability-flow ODE** for fast, deterministic sampling.
- **SDE solvers** with adaptive step sizes.

## Conditional generation

For text-to-image, condition the noise predictor on text embedding $c$: $\epsilon_\theta(x_t, t, c)$.

### Classifier-free guidance (Ho & Salimans 2022)

The most important trick in modern diffusion. Train one network jointly on conditional and unconditional inputs (with random "null" conditioning). At sample time:
$$\hat\epsilon_\theta(x_t, t, c) = (1 + w) \epsilon_\theta(x_t, t, c) - w\, \epsilon_\theta(x_t, t, \emptyset).$$

The guidance weight $w$ trades **diversity for prompt faithfulness**:
- $w = 0$: standard conditional.
- $w \in [3, 10]$ typical for text-to-image — sharper, more on-prompt.
- High $w$: over-saturated, unnatural.

## Latent diffusion (Rombach et al. 2022, "Stable Diffusion")

Diffuse not in pixel space but in a learned **VAE latent space** (typically 4× downsampled):
1. VAE encoder: $x \to z$ (in $\mathbb{R}^{H/8 \times W/8 \times 4}$ for $H \times W$ images).
2. Diffusion in $z$-space: 64× fewer parameters needed.
3. VAE decoder: $z \to x$ at the end.

Makes high-resolution generation tractable. Stable Diffusion runs on consumer GPUs because of this.

## Architectures

Standard backbone: **U-Net** with self-attention layers at lower resolutions, conditioned on time $t$ via sinusoidal embeddings + an MLP.

Modern variants:
- **DiT — Diffusion Transformer** (Peebles & Xie 2022). Transformer instead of U-Net. SOTA for image (DALL-E 3) and video (Sora).
- **AlphaFold 3** uses a diffusion-style module for protein structure (3D atomic coordinates).

## Why diffusion won

1. **Stable training** — just MSE, no adversarial dynamics.
2. **Mode coverage** — generates diverse samples (vs. GAN mode collapse).
3. **Scales gracefully** with compute and parameters.
4. **Theoretical grounding** — score matching, SDE/ODE, variational bounds.
5. **Conditional generation works really well** — classifier-free guidance is a remarkably simple yet powerful trick.

## What diffusion doesn't do well

- **Slow inference** — even 10-step samplers are slower than 1-step GANs.
- **Exact likelihood** is intractable (only bounds).
- **Continuous-only.** Discrete data (text, code) needs alternatives — but **discrete diffusion** is an active research area, getting closer.

## See also

- [[VAEs]] — latent diffusion uses a VAE for compression.
- [[GANs]] — what diffusion replaced.
- [[Transformer]] — modern diffusion backbones (DiT) are transformers.
- [[Optimisation]] — Adam(W) + warmup + cosine; large EMA on weights (decay ≈ 0.9999).
