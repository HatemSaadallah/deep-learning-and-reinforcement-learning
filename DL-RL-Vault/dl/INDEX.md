# Deep Learning — folder index

Navigate the DL side of the vault. Most notes are hub-style — one per topic, with the canonical formulas, the key intuition, and pointers to the others.

## Foundations

- [[Neural Network Fundamentals]] — linear regression, shallow/deep nets, backprop, loss functions, MLE, UAT
- [[Optimisation]] — SGD, momentum, Adam(W), schedules, loss landscape
- [[Initialisation]] — Xavier/He, vanishing/exploding gradients
- [[Regularisation]] — L1/L2, dropout, BatchNorm/LayerNorm, data aug, implicit
- [[Overparameterisation]] — double descent, NTK, lottery tickets, scaling laws

## Architectures

- [[Transformer]] — self-attention, multi-head, positional encoding, encoder/decoder
- [[GNNs]] — message passing, GCN/GAT/GIN, over-smoothing
- [[Transformers Biology]] — ESM, AlphaFold, DNA/RNA models

## Generative models

- [[VAEs]] — ELBO, reparameterisation trick, β-VAE, VQ-VAE
- [[GANs]] — generator/discriminator, mode collapse, WGAN, StyleGAN
- [[Diffusion Models]] — DDPM, score-based, latent diffusion, classifier-free guidance

## Practical / mixed

- [[Mixed DL Techniques]] — activation table, training recipe, mixed precision, gradient clipping, debugging checklist
- [[Unsupervised Learning]] — pretraining paradigms (MLM, autoregressive, contrastive, BYOL/SimSiam, CLIP)

## Cross-references to the RL side of the vault

| DL topic | Related RL topic |
|---|---|
| [[Optimisation]] (loss landscape) | [[OGD Regret Bound]] (online setting) |
| [[Regularisation]] (KL terms) | [[KL Divergence]], [[RLHF and DPO]] (KL penalty) |
| [[Transformer]] | [[RLHF and DPO]] (RL aligns transformer LLMs) |
| [[Optimisation]] (Adam) | [[Q-learning]] (gradient methods for Q-net) |
| [[GANs]] (zero-sum game) | [[Regret and Equilibria]] (no-regret → Nash) |

The DL and RL parts share more theory than is obvious at first — Adam is used for everything, KL divergence shows up in both ELBO and TRPO, and game-theoretic intuitions from GANs reappear in adversarial RL.

## Conventions

- Math: $\phi$ for NN parameters, $x$ for input data, $y$ for targets, $\hat y$ for predictions, $h^{(\ell)}$ for hidden activation at layer $\ell$.
- Loss: $\mathcal{L}$ or $L$, summed/averaged over data.
- "Activation function" = the non-linearity ($\sigma$, ReLU, GELU). "Activation" = the vector $h$ after applying it.
