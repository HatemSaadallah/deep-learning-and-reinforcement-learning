# Graph Neural Networks (GNNs)

**Tags:** #dl #architecture #graphs

Neural networks designed for **graph-structured data**: nodes connected by edges, where the connectivity is part of the input. Used for molecules, social networks, knowledge graphs, recommender systems, traffic, protein interaction, etc.

## The setting

A graph $G = (V, E)$ with:
- Nodes $v \in V$ with feature vectors $x_v \in \mathbb{R}^{d_{\text{node}}}$.
- Edges $(u, v) \in E$, possibly with features $e_{uv} \in \mathbb{R}^{d_{\text{edge}}}$.
- Possibly directed, possibly with self-loops.

Tasks:
- **Node-level:** predict label per node (e.g. user demographics).
- **Edge-level:** predict link existence/weight (e.g. recommendations).
- **Graph-level:** predict label per graph (e.g. molecule toxicity).

## Message passing — the unified framework

Almost every GNN can be described as $T$ rounds of **message passing** (Gilmer et al. 2017):

For each round $t = 1, \dots, T$:
1. **Message:** $m_{uv}^{(t)} = M_t(h_u^{(t-1)}, h_v^{(t-1)}, e_{uv})$
2. **Aggregate:** $a_v^{(t)} = \mathrm{AGG}\bigl(\{m_{uv}^{(t)} : u \in \mathcal{N}(v)\}\bigr)$ (sum, mean, max, attention-weighted)
3. **Update:** $h_v^{(t)} = U_t(h_v^{(t-1)}, a_v^{(t)})$

After $T$ rounds, $h_v^{(T)}$ encodes information from the $T$-hop neighborhood of $v$.

For graph-level tasks, **pool** the final node embeddings: $h_G = \mathrm{POOL}(\{h_v^{(T)} : v \in V\})$ (sum, mean, attention).

## Canonical GNN architectures

### GCN — Graph Convolutional Network (Kipf & Welling 2017)

$$H^{(t)} = \sigma\bigl(\tilde D^{-1/2}\, \tilde A\, \tilde D^{-1/2}\, H^{(t-1)} W^{(t)}\bigr)$$
where $\tilde A = A + I$ (add self-loops), $\tilde D$ is its degree matrix. Each node aggregates a **degree-normalised** average of neighbour features.

Simple, fast, surprisingly competitive baseline.

### GraphSAGE (Hamilton et al. 2017)

Sampled aggregation — for large graphs, sample a fixed-size neighbourhood rather than process all neighbours. Aggregator can be mean, LSTM, max pool. Inductive: works on unseen nodes.

### GAT — Graph Attention Network (Veličković et al. 2018)

Attention-weighted aggregation:
$$\alpha_{uv} = \mathrm{softmax}_u\!\bigl(\mathrm{LeakyReLU}(a^\top [W h_u \| W h_v])\bigr)$$
$$h_v^{(t)} = \sigma\!\Bigl(\sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u^{(t-1)}\Bigr)$$
Multi-head extension. Better expressive power than GCN.

### GIN — Graph Isomorphism Network (Xu et al. 2019)

The **most expressive message-passing GNN** (as powerful as the Weisfeiler-Lehman graph isomorphism test).
$$h_v^{(t)} = \mathrm{MLP}^{(t)}\!\Bigl((1+\epsilon^{(t)}) h_v^{(t-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(t-1)}\Bigr).$$
Sum aggregation > mean > max in terms of expressivity.

## Fundamental limit: 1-WL

Message-passing GNNs are **at most as powerful as the 1-Weisfeiler-Lehman (1-WL) test**: they can't distinguish graphs that 1-WL can't distinguish. Examples of indistinguishable pairs: two cycles vs. one larger cycle of the same total node count.

To go beyond 1-WL:
- **Higher-order GNNs** (k-GNN): operate on $k$-tuples of nodes.
- **Augment node features** with structural fingerprints (degree, centrality, random IDs).
- **Subgraph GNNs**: run GNN on a bag of subgraphs.
- **Graph transformers**: full self-attention over all nodes (loses inductive bias of sparsity).

## Over-smoothing

Stacking many GNN layers causes node embeddings to **converge to the same vector** — the network can't distinguish nodes anymore. Fundamental issue: each round mixes info with neighbours, eventually mixing across the entire connected component.

**Mitigations:**
- **Residual / skip connections** (like in ResNet).
- **Identity initialisation** with edge dropping.
- **Pair-norm / GraphNorm**: normalisation tailored to graphs.
- Use few layers (2–4) — works fine for most tasks.

## Graph transformers

A growing alternative: apply [[Transformer|transformer]] self-attention to all pairs of nodes, with **positional encodings derived from graph structure** (Laplacian eigenvectors, random walks).

Examples: GraphGPS, Graphormer, GraphiT. Strong empirical results on large graphs, especially molecules.

Trade-off: $O(|V|^2)$ attention cost, but escapes 1-WL barrier.

## Applications

| Domain | Example |
|---|---|
| Chemistry | Molecule property prediction (ZINC, OGB), drug discovery, MoleculeNet |
| Materials science | Crystal property prediction |
| Drug-target interaction | Predict binding affinity |
| Recommender systems | PinSage (Pinterest), LightGCN |
| Knowledge graphs | Link prediction, entity classification |
| Physics simulation | Mesh-based fluid dynamics, particle systems |
| Code | Bug detection, completion (control flow graphs) |
| Maps / routing | ETA prediction on road networks (Google Maps uses GNNs) |

## Practical notes

- **Datasets:** OGB (Open Graph Benchmark) is the de facto standard for fair comparison.
- **Library:** PyTorch Geometric (PyG) or DGL.
- **Pooling for graph-level tasks:** sum is theoretically best (preserves WL info), mean / attention in practice.
- **Negative sampling** for link prediction is crucial — sampling strategy matters.

## See also

- [[Transformer]] — graph transformers borrow from this directly.
- [[Optimisation]] — Adam, but smaller learning rates than for transformers ($10^{-4}$).
- [[Unsupervised Learning]] — graph contrastive learning is a self-supervised pretraining trick.
