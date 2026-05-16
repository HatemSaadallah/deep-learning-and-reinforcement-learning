# Transformer

**Tags:** #dl #architecture #foundational

The architecture that displaced RNNs and CNNs as the universal sequence model. Introduced by Vaswani et al. 2017 ("Attention is All You Need"). Backbone of every modern LLM (GPT, Llama, Claude, DeepSeek), most vision models (ViT, Swin), and increasingly bio/protein/audio models.

## The big idea

Replace recurrence and convolution with **self-attention** — every token in the sequence directly attends to every other token. Two consequences:
- **Parallel computation:** unlike RNNs, no sequential dependency in training (just a giant matrix multiplication per layer).
- **Long-range dependencies:** any pair of tokens is one "hop" apart, no information bottleneck.

## Scaled dot-product attention

Given query $Q$, key $K$, value $V$ matrices (all of shape $L \times d$ where $L$ is sequence length):
$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.$$

**Reading:**
- $QK^\top \in \mathbb{R}^{L \times L}$ — for each query, similarity to each key.
- $/\sqrt{d_k}$ — prevents softmax saturation in high $d_k$ (variance argument).
- $\mathrm{softmax}(\cdot)$ — converts similarities to a probability distribution over keys.
- Multiply by $V$ — output is a weighted average of value vectors.

In **self-attention**, $Q, K, V$ are all linear projections of the same input $X$: $Q = XW_Q$, $K = XW_K$, $V = XW_V$.

## Multi-head attention

Run $h$ parallel attention "heads" of dimension $d_k = d_{\text{model}} / h$, concatenate, project:
$$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,W_O.$$

Each head can specialize: some look at syntax, some semantics, some long-range patterns. Empirically essential — single-head attention underperforms.

## A transformer block

```
input X
   │
   ├──→ LayerNorm → Multi-Head Self-Attention ──┐
   │                                            │  (residual)
   X' = X + ─────────────────────────────────── ┘
   │
   ├──→ LayerNorm → FeedForward Network ────────┐
   │                                            │  (residual)
   X'' = X' + ────────────────────────────────  ┘
   │
   output X''
```

**Feed-forward network (FFN):** two-layer MLP with activation (GELU or SwiGLU in modern models):
$$\mathrm{FFN}(x) = W_2 \cdot \mathrm{GELU}(W_1 x + b_1) + b_2.$$
Inner dimension $d_{\text{ff}} \approx 4 d_{\text{model}}$ typically.

**Residual connections** allow gradient flow and stable training of very deep stacks. **Pre-LN** (LayerNorm before the sublayer, as drawn above) is the modern standard — more stable than the original post-LN.

## Positional encoding

Attention is **permutation-equivariant** (doesn't know order). Inject position via positional encodings added to (or fused with) token embeddings.

| Type | Use |
|---|---|
| **Sinusoidal** (original) | fixed, extrapolates to longer sequences |
| **Learned absolute** | per-position vectors, trained |
| **Relative position** (Shaw et al.) | bias attention scores by $i-j$ |
| **RoPE** (Su et al. 2021) | rotate Q, K by position-dependent angle — used in Llama, GPT-NeoX, DeepSeek |
| **ALiBi** (Press et al. 2022) | additive linear bias on attention scores |

RoPE has become the de facto modern default.

## Causal (masked) self-attention

For autoregressive language modeling, prevent token $t$ from attending to tokens $> t$ by setting those entries of $QK^\top$ to $-\infty$ before softmax:
$$\mathrm{mask}_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}$$

This makes training **parallelizable across the sequence** — for each position $t$, you compute the loss assuming tokens $1, \dots, t$ are visible, and gradients flow normally — but at inference time generation is **sequential**.

## Architecture variants

| | Used in | Description |
|---|---|---|
| **Encoder-only** | BERT, RoBERTa, ViT | bidirectional attention; for representation / classification |
| **Decoder-only** | GPT family, Llama, DeepSeek | causal attention; for generation |
| **Encoder-decoder** | T5, BART, original Transformer (translation) | encoder reads source, decoder generates target with cross-attention |

The decoder-only architecture has won the LLM era — same forward pass for pretraining (next-token prediction) and downstream tasks.

## Complexity

| Operation | Cost |
|---|---|
| Self-attention | $O(L^2 \cdot d)$ — **quadratic in sequence length** |
| FFN | $O(L \cdot d \cdot d_{\text{ff}})$ |
| Linear projections | $O(L \cdot d^2)$ |

The $L^2$ scaling of attention is the **dominant cost** for long sequences. Hence the explosion of "efficient transformer" research:
- **FlashAttention** (Dao et al. 2022): reorders the attention computation for IO efficiency — same exact computation, much faster.
- **Sparse / sliding-window attention**: only attend to a subset of positions (Longformer, BigBird, Mistral).
- **Linear attention**: replace softmax with a kernel that factorises (Performers, Linformer). Trades quality for speed.
- **Mamba / S4** (state-space models): subquadratic alternative, gaining traction.

## Training tricks

- **Pre-LN > Post-LN** for stability of very deep models.
- **RMSNorm > LayerNorm** in modern LLMs (simpler, slightly cheaper).
- **SwiGLU** activation in FFN beats GELU.
- **No biases** in many recent architectures (Llama-style).
- **Untying input/output embeddings** for very large models.

## Why transformers work so well

1. **Massive parallelism** — full sequence in one matmul, exploits GPU well.
2. **Soft inductive bias** — no hard architectural assumption about locality (vs. CNNs) or sequentiality (vs. RNNs); attention learns whatever patterns matter.
3. **Composability** — stacking many blocks is well-behaved with residuals + LN; very deep networks work.
4. **Scales smoothly** with parameters and data — [[Overparameterisation|scaling laws]] are clean.

## See also

- [[Optimisation]] — Adam(W) + warmup + cosine schedule is the standard recipe.
- [[Initialisation]] — transformers need careful init (T-Fixup, layer scaling).
- [[Regularisation]] — LayerNorm/RMSNorm in every block.
- [[Transformers Biology]] — applications in protein structure, genomics.
- [[RLHF and DPO]] (RL vault) — how LLMs are fine-tuned post-pretraining.
