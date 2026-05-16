# Yao's Principle

**Tags:** #lower-bounds #minimax #foundational

"Any randomized algorithm can be expressed as a distribution over deterministic algorithms."

## Why it's useful

Lower bounds are usually easier to prove against **deterministic** algorithms (you fix the algorithm, then construct a bad input). Yao lets you lift such a bound to randomized algorithms automatically.

## Formal version

For any randomized algorithm $\mathcal{A}$ and input $x$:
$$\max_x \mathbb{E}_{\mathcal{A}}[\text{cost}(\mathcal{A}, x)] \geq \min_{\text{det } A} \mathbb{E}_{x \sim \mu}[\text{cost}(A, x)]$$
for any input distribution $\mu$. Pick $\mu$ adversarially to maximize the RHS.

## Where it shows up

- [[Bandit Lower Bound]] — extends the deterministic lemma to randomized bandit algorithms.
- Online algorithm lower bounds.
- Communication complexity, data-structure lower bounds.

## Application in bandits

Lemma says: for a *deterministic* algorithm and $T \leq cK/\epsilon^2$, there's an instance with failure prob $\geq 1/4$. Yao: same holds for any *randomized* algorithm against the uniform-over-instances distribution.
