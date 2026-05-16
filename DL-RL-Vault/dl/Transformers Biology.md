# Transformers in Biology

**Tags:** #dl #applications #biology

[[Transformer|Transformer architectures]] applied to biological sequences and structures. Two flagship breakthroughs: **AlphaFold** (protein structure prediction) and **ESM** family (protein language models).

## Why transformers are a fit for biology

Biological data is often sequence-shaped (DNA, RNA, proteins) with strong long-range dependencies — exactly what attention is designed for. Properties:
- DNA: 4-letter alphabet (A, C, G, T), sequences from $\sim 10^3$ to $10^9$ bases.
- RNA: 4-letter alphabet (A, C, G, U), sequences typically $10^2$ to $10^4$.
- Proteins: 20-letter alphabet (amino acids), sequences typically $10^2$ to $10^4$.

The "tokens" of biology have rich semantics (chemistry, function) much like words in natural language — and crucially, **lots of unlabeled sequence data** exists (UniProt, GenBank), enabling pretraining.

## Protein language models (PLMs)

Train a transformer with masked language modelling on millions of protein sequences. The model learns:
- Co-evolution patterns (residues that mutate together → likely in physical contact)
- Conserved motifs
- Structural / functional features

### ESM family (Meta AI)

- **ESM-1b** (Rives et al. 2021): 650M params, MLM on UniRef50.
- **ESM-2** (Lin et al. 2023): up to 15B params; pretraining alone produces representations that **predict 3D structure** with one MLP head.
- **ESMFold**: ESM-2 + structure head → fast (10x ESM-1v) protein structure prediction without MSA.

**Key takeaway:** A pure language model on protein sequences (no labels, no MSAs) recovers structural and evolutionary signal as a side effect of next-token / masked-token prediction.

## AlphaFold (DeepMind)

Not strictly "a transformer" — uses **Evoformer** (a transformer-like module operating jointly on sequence and pair representations) and a structure module.

| | AlphaFold 2 | AlphaFold 3 |
|---|---|---|
| Year | 2021 | 2024 |
| Input | MSA + template | MSA + template + ligands |
| Output | protein structure | proteins, nucleic acids, small molecules, **complexes** |
| Architecture | Evoformer + structure module | diffusion-based structure decoder |

AF2 essentially **solved single-protein structure prediction** for sequences with available homologs.

## DNA and genomics

- **DNABERT** (Ji et al. 2021): BERT on DNA k-mers.
- **Nucleotide Transformer** (InstaDeep / NVIDIA): 2.5B params, multi-species DNA.
- **HyenaDNA, Caduceus**: state-space models for long DNA (up to 1M tokens) since attention is too expensive at genome scale.

Tasks: enhancer prediction, splice site prediction, transcription factor binding.

## RNA

- **RNA-FM, RNA-MSM**: protein-LM-style approaches transferred to RNA.
- Challenge: RNA secondary structure (base pairs) is more important than for proteins, encoded poorly by pure sequence.

## Cell biology / single-cell

- **scGPT, Geneformer**: transformers over gene expression vectors of individual cells (tokens = genes ranked by expression).
- Task: cell type annotation, perturbation prediction, drug response.

## Multimodal biological models

Combining sequence + structure + image + clinical data. Emerging area:
- **AlphaFold 3** with ligands/RNA/DNA.
- **scFoundation** / **scGPT** for spatial transcriptomics.
- **PLIP / MedSAM** for histopathology images + text.

## Common recipe

```
pretrain on huge unlabeled biological corpus (MLM or autoregressive)
   ↓
fine-tune for downstream task (binding affinity, structure, function)
or
linear probe on frozen embeddings
```

Pretraining recipe matches NLP almost exactly. Fine-tuning often uses LoRA or adapter layers because labelled biological data is scarce.

## Why this works (and where it doesn't)

**Works well:**
- Single proteins (sequence-only) — AF2, ESM-2.
- Genomic motif prediction.
- Cell type identification.

**Works less well:**
- Protein-protein dynamics, conformational ensembles.
- Allostery and long-time-scale dynamics.
- Truly novel folds (proteins outside training distribution).
- De novo design without iteration with experimental feedback.

## See also

- [[Transformer]] — the underlying architecture.
- [[Unsupervised Learning]] — masked language modelling is the standard pretraining objective.
- [[Diffusion Models]] — AlphaFold 3 and modern protein-design tools (RFdiffusion) use diffusion in 3D coordinate space.
