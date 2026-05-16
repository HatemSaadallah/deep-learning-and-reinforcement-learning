# Unsupervised / Self-supervised Learning

**Tags:** #dl #framework

Learning useful representations from **data without labels**. The dominant paradigm of modern deep learning: pretrain on massive unlabeled corpora (the web, ImageNet, UniProt) via a self-supervised objective, then fine-tune for specific tasks.

## Unsupervised vs. self-supervised

The terms blur, but a common convention:
- **Unsupervised:** no labels at all. Goals: clustering, density estimation, dimensionality reduction.
- **Self-supervised:** the input itself provides a "free" label via some pretext task (mask a token, predict the next pixel).

Almost all modern "unsupervised learning" in deep learning is actually self-supervised.

## Classical unsupervised baselines

| Method | Idea |
|---|---|
| **PCA** | Linear projection to top-$k$ principal components. |
| **k-means** | Cluster into $k$ groups by Euclidean distance. |
| **Gaussian Mixture Model** | $k$ Gaussians fit via EM. Soft clustering. |
| **t-SNE / UMAP** | Non-linear dim reduction for visualisation (preserves neighbourhoods). |
| **Autoencoders** | NN that compresses to bottleneck and reconstructs. |

These remain useful for low-dim or labeled data, but **don't scale to deep representation learning** the way self-supervised does.

## The pretrain → fine-tune paradigm

```
Stage 1: PRETRAIN on huge unlabeled data
   ↓
   produces a network with strong general representations
   ↓
Stage 2: FINE-TUNE on small labeled task data
   ↓
   adapt to the specific task; small labels suffice
```

This decouples the "expensive part" (representation learning) from the "task-specific part" (calibrating to labels). The single most important methodological shift of deep learning in the past decade.

## Self-supervised objectives

### Reconstruction-based

**Autoencoder:** encode to bottleneck $z$, decode to $\hat x$, minimize $\|x - \hat x\|$. Bottleneck forces a compressed representation. [[VAEs]] add a probabilistic twist.

**Denoising autoencoder:** corrupt input ($x + \text{noise}$), reconstruct clean $x$. The denoising forces the network to learn the data manifold (key insight: this is what [[Diffusion Models]] generalise).

### Masked prediction (the BERT / MAE family)

**BERT** (NLP, Devlin et al. 2019):
- Randomly mask 15% of tokens.
- Train transformer to predict the masked tokens from context.
- Bidirectional → learns rich representations but can't generate naively.

**MAE — Masked Autoencoder for vision** (He et al. 2021):
- Mask 75% of image patches.
- Lightweight encoder on visible patches → heavy decoder reconstructs missing pixels.
- Strong vision representations.

**MLM (Masked Language Modelling)** is the dominant pretraining objective for encoder transformers.

### Autoregressive (the GPT family)

Predict next token given prefix:
$$\max_\theta \sum_t \log p_\theta(x_t \mid x_{<t}).$$

Standard next-token prediction. Unlike masked LM, the same forward pass works for generation. Powers the entire LLM era.

### Contrastive learning

Bring similar examples together in embedding space, push dissimilar apart.

**SimCLR** (Chen et al. 2020): two augmentations of the same image are a "positive pair", augmentations of other images are negatives. InfoNCE loss:
$$\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(z_i, z_i^+) / \tau)}{\sum_j \exp(\mathrm{sim}(z_i, z_j) / \tau)}.$$

**MoCo** (He et al. 2020): same idea + a momentum-updated key encoder + a queue of negatives.

**DINO** (Caron et al. 2021): teacher-student self-distillation, no negatives. Emergent properties (attention maps localize objects without labels!).

**CLIP** (Radford et al. 2021): contrastive learning between **image-text pairs**. Foundation of multimodal models.

### Predictive / latent

**BYOL** (Grill et al. 2020): online network predicts target network output. No negatives. Avoided collapse via stop-gradient + EMA target.

**SimSiam** (Chen & He 2021): even simpler, no momentum target. Stop-gradient is the secret.

## Generative vs. discriminative pretraining

| | Discriminative (contrastive, BYOL, SimCLR) | Generative (MLM, autoregressive, MAE) |
|---|---|---|
| Goal | distinguish from negatives | reconstruct / predict tokens |
| Better for | classification, retrieval | generation, language modelling |
| Compute | usually less per epoch | more (decoder cost) |
| Sample efficiency | high | moderate |

In practice the lines blur — most state-of-the-art systems combine multiple objectives.

## The "foundation model" era

Pretrained models at scale that serve as bases for downstream tasks:
- **GPT-4, Claude, Llama 3, DeepSeek**: text + multimodal LLMs.
- **DINOv2, MAE, SAM**: vision foundations.
- **ESM-2, AlphaFold-3**: biology foundations.
- **Whisper**: audio foundation.

The same recipe — massive unlabeled corpus + self-supervised objective + scale — works in every modality.

## See also

- [[Transformer]] — the architecture all of this runs on.
- [[VAEs]] — probabilistic latent variable model.
- [[GANs]] — generative model without explicit likelihood.
- [[Diffusion Models]] — current SOTA generative method.
- [[RLHF and DPO]] (RL vault) — how pretrained LMs are aligned after self-supervised pretraining.
