# Mixed DL Techniques (Teaching notes)

**Tags:** #dl #practical

Grab-bag of "things you actually do" when training deep nets in practice. Covers points often spread across multiple papers/lectures that all matter for getting a model to work.

## Activation functions

| Function | Formula | Pros / Cons |
|---|---|---|
| **ReLU** | $\max(0, x)$ | Cheap, sparse activations, simple gradients. Dying-ReLU problem (neurons stuck at 0). De facto default. |
| **Leaky ReLU** | $\max(\alpha x, x)$, $\alpha = 0.01$ | Fixes dying-ReLU; rarely worth it over ReLU. |
| **ELU / SELU** | $x$ if $> 0$, else $\alpha(e^x - 1)$ | Mean-zero outputs, self-normalising under conditions. Niche. |
| **GELU** | $x \cdot \Phi(x)$ (Gaussian CDF) | Smooth ReLU. Default in transformers (BERT, GPT-2). |
| **SwiGLU** | $\mathrm{Swish}(xW_1) \odot xW_2$ | Modern transformer default (Llama, DeepSeek). Beats GELU by ~1% on benchmarks. |
| **Tanh / Sigmoid** | $\tanh(x)$, $\sigma(x)$ | Outputs bounded; vanishing gradients at saturation. Used for output of binary classifiers, RNN gates. |

## Loss functions

| Task | Standard loss |
|---|---|
| Regression | MSE $\|y - \hat y\|^2$, or Huber for robustness |
| Binary classification | Binary cross-entropy $-y \log\hat p - (1-y)\log(1-\hat p)$ |
| Multi-class classification | Categorical cross-entropy $-\sum_c y_c \log p_c$ |
| Imbalanced classes | Focal loss $-(1-p_t)^\gamma \log p_t$ |
| Distance/similarity | Triplet loss, contrastive (InfoNCE) |
| Segmentation | Cross-entropy + Dice |
| Object detection | Smooth-L1 (boxes) + cross-entropy (class) |
| Sequence generation | Next-token cross-entropy (with label smoothing) |

## Normalisation choices

Already covered in [[Regularisation]]. Cheat sheet:
- **BatchNorm** — CNNs, large batches.
- **LayerNorm** — transformers, RNNs, small/variable batches.
- **RMSNorm** — modern LLMs (Llama-style).
- **GroupNorm** — small batches (e.g. detection, video).

## Practical training recipe (modern transformer)

```
optimizer:        AdamW
   betas:         (0.9, 0.95) for LLMs, (0.9, 0.999) standard
   weight_decay:  0.1 (LLM) or 0.01-0.001 (CV)
   eps:           1e-8
learning_rate:    peak 1e-4 to 3e-4 (LLM), 1e-3 (CV)
schedule:         warmup (1-5% of steps) + cosine decay to 10% of peak
batch_size:       global 1M-4M tokens for LLMs; 1024-4096 for CV
grad_clip:        norm 1.0
mixed_precision:  bf16 or fp16 with loss scaling
EMA:              decay 0.9999 for diffusion models, optional for LLMs
gradient_accum:   when memory-limited
```

## Mixed-precision training

Train in float16 / bfloat16 to halve memory and double throughput.
- **fp16**: needs **dynamic loss scaling** to avoid gradient underflow. Tricky.
- **bf16**: same range as fp32, much easier. Standard on modern (A100+, H100, TPU).

## Gradient clipping

Prevent gradient explosions:
$$g \leftarrow g \cdot \min\bigl(1, \tfrac{c}{\|g\|}\bigr).$$
Standard threshold $c = 1.0$ for transformers. Essential for stability of large models.

## Curriculum / data scheduling

Order in which data is presented can matter:
- **Easy → hard** examples (classical curriculum learning).
- **Length-based bucketing** for transformers — group similar-length sequences together for compute efficiency.
- **Data mixture tuning** for LLMs — balance code, web text, books, etc.

## Hyperparameter intuition

Five "knobs" that most affect outcomes (in order of importance):
1. **Learning rate** — by far the most sensitive.
2. **Batch size** — interacts with LR (linear scaling rule).
3. **Number of training steps / epochs**.
4. **Weight decay**.
5. **Warmup length**.

Other hyperparameters (dropout rate, layer norm position, etc.) matter much less.

## Compute optimisation

- **Use FlashAttention** if attention is the bottleneck.
- **Activation checkpointing** trades compute for memory (recompute activations during backward pass).
- **Gradient accumulation** simulates larger batch size with limited memory.
- **Tensor / pipeline / data parallelism** for multi-GPU.
- **ZeRO / FSDP** for memory-efficient distributed training.

## Debugging a non-training network

Classic checklist:
1. **Overfit a tiny batch** (e.g. 1 sample). If you can't, something is fundamentally broken.
2. **Check loss is decreasing** in the first 100 steps.
3. **Inspect gradients** — too small? exploding? NaN?
4. **Check init** — are weights too small/large?
5. **Verify data loading** — labels match inputs? Augmentations correct?
6. **Sanity-check with simpler model** — if a logistic regression doesn't work on the data, the deep model won't either.

## Reproducibility

- Set seeds for python `random`, NumPy, PyTorch (`manual_seed`).
- Use `torch.backends.cudnn.deterministic = True` (slower but reproducible).
- Log all hyperparameters and code version.
- Beware: distributed training, mixed precision, and some CUDA ops are inherently non-deterministic. "Bit-exact" reproducibility is hard.

## See also

- [[Optimisation]] — gradient-descent variants.
- [[Initialisation]] — get the start right.
- [[Regularisation]] — keep the training honest.
- [[Transformer]] — most modern practical recipes are transformer-centric.
