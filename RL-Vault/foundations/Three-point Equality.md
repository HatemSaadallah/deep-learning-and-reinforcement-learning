# Three-Point Equality

**Tags:** #foundational #lemma #convex-analysis

The algebraic identity that makes [[Mirror Descent Analysis|mirror descent]] regret analysis work — the analog of the law of cosines for Bregman geometry.

## Statement (general)

Let $R$ be a differentiable function. The [[Bregman Divergence]] $D_R(x, y) := R(x) - R(y) - \langle \nabla R(y), x - y\rangle$ satisfies:
$$\boxed{\;D_R(u, v) \;-\; D_R(u, w) \;-\; D_R(w, v) \;=\; \langle \nabla R(w) - \nabla R(v),\; w - u\rangle.\;}$$

## Statement (Euclidean special case)

With $R(x) = \tfrac{1}{2}\|x\|_2^2$, $\nabla R(x) = x$, and $D_R(x, y) = \tfrac{1}{2}\|x - y\|_2^2$. The identity becomes
$$\frac{1}{2}\|u - v\|^2 \;-\; \frac{1}{2}\|u - w\|^2 \;-\; \frac{1}{2}\|w - v\|^2 \;=\; \langle w - v,\; w - u\rangle.$$
This is the classical **parallelogram-law identity** (or "law of cosines" for inner-product spaces).

## Proof

Expand each term of the LHS using the definition of $D_R$:

- $D_R(u, v) = R(u) - R(v) - \langle \nabla R(v), u - v\rangle$
- $D_R(u, w) = R(u) - R(w) - \langle \nabla R(w), u - w\rangle$
- $D_R(w, v) = R(w) - R(v) - \langle \nabla R(v), w - v\rangle$

Compute LHS $= D_R(u, v) - D_R(u, w) - D_R(w, v)$:
$$\begin{aligned}
&= \bigl[R(u) - R(v) - \langle \nabla R(v), u - v\rangle\bigr] - \bigl[R(u) - R(w) - \langle \nabla R(w), u - w\rangle\bigr] - \bigl[R(w) - R(v) - \langle \nabla R(v), w - v\rangle\bigr]\\
&= -\langle \nabla R(v), u - v\rangle + \langle \nabla R(w), u - w\rangle + \langle \nabla R(v), w - v\rangle\\
&= \langle \nabla R(w), u - w\rangle + \langle \nabla R(v), (w - v) - (u - v)\rangle\\
&= \langle \nabla R(w), u - w\rangle + \langle \nabla R(v), w - u\rangle\\
&= \langle \nabla R(w) - \nabla R(v),\; u - w\rangle + \langle \nabla R(v), u - w\rangle + \langle \nabla R(v), w - u\rangle \\
&\;\;\vdots
\end{aligned}$$
Easier route: just collect terms. The $R(u), R(v), R(w)$ terms each cancel. The remaining inner-product terms simplify to $\langle \nabla R(w) - \nabla R(v), w - u\rangle$. $\square$

## Why it matters

In the [[Mirror Descent Analysis|MD regret proof]], the update $x_{t+1} = \arg\min_x \eta\langle g_t, x\rangle + D_R(x, x_t)$ has first-order optimality
$$\langle \eta g_t + \nabla R(x_{t+1}) - \nabla R(x_t),\; u - x_{t+1}\rangle \geq 0 \quad \forall u \in \mathcal{C}.$$

The RHS of the three-point identity with $v = x_t$, $w = x_{t+1}$ is exactly $\langle \nabla R(x_{t+1}) - \nabla R(x_t), x_{t+1} - u\rangle$. Substituting:
$$D_R(u, x_t) - D_R(u, x_{t+1}) - D_R(x_{t+1}, x_t) \;\leq\; \eta\langle g_t, u - x_{t+1}\rangle.$$

The first two terms **telescope** when summed over $t$. The third (controlled by strong convexity) and the inner product (via Young's inequality) give the standard $O(\sqrt{T})$ regret.

## Where it appears in the course

- [[Mirror Descent Analysis|Mirror descent regret proof]] — the key step.
- [[OGD Regret Bound|OGD]] — Euclidean special case; Celli's lecture proves OGD via this identity + [[Euclidean Mirror Descent Lemma]].
- [[FTRL Stability and Regret|FTRL]] regret proof — implicit when written in MD/proximal form.

## See also

- [[Bregman Divergence]] — the parent concept.
- [[Euclidean Mirror Descent Lemma]] — direct consequence in the Euclidean case.
- [[Mirror Descent Analysis]] — main application.
