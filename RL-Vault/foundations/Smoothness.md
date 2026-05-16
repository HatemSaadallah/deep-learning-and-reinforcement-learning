# Smoothness ($L$-smooth functions)

**Tags:** #foundational #convex-analysis #definition

The dual concept to [[Convexity|strong convexity]]: instead of curving up *at least as fast* as a quadratic, an $L$-smooth function curves up *at most as fast* as a quadratic.

## Definition

A differentiable function $f: \mathcal{D} \to \mathbb{R}$ is **$L$-smooth** for $L > 0$ if for every $x, y \in \mathcal{D}$:
$$f(y) \;\leq\; f(x) + \langle \nabla f(x),\, y - x\rangle + \frac{L}{2}\|y - x\|_2^2.$$

The **smaller $L$**, the smoother. The gradient grows slowly.

## Alternative definition (Lipschitz gradient)

$f$ is $L$-smooth iff $\nabla f$ is $L$-Lipschitz:
$$\|\nabla f(x) - \nabla f(y)\|_2 \leq L\|x - y\|_2 \qquad \forall x, y.$$

"Gradients don't change much when you move slightly."

## Geometric reading

- A general $L$-smooth function admits an **upper quadratic bound** at every point.
- An $L$-smooth **convex** function is sandwiched between a lower linear bound (from convexity) and an upper quadratic bound (from smoothness). Like a hamburger.

## Smooth vs. strongly convex

| | Bound type | Meaning |
|---|---|---|
| $\mu$-strongly convex | $f(y) \geq f(x) + \langle \nabla f(x), y-x\rangle + \tfrac{\mu}{2}\|y-x\|^2$ | curves up at least $\mu/2 \cdot$ quadratic |
| $L$-smooth | $f(y) \leq f(x) + \langle \nabla f(x), y-x\rangle + \tfrac{L}{2}\|y-x\|^2$ | curves up at most $L/2 \cdot$ quadratic |
| Both | $\mu \leq \nabla^2 f \preceq L$ (if $C^2$) | condition number $\kappa = L/\mu$ |

## Examples

| Function | Smooth? | $L$ |
|---|---|---|
| $\tfrac{1}{2}\|x\|^2$ | yes | $1$ |
| $\|x\|^2$ | yes | $2$ |
| $\log(1 + e^x)$ (softplus) | yes | $1/4$ |
| Squared loss $(y - \langle\theta, x\rangle)^2$ | yes | $2\|x\|_2^2$ |
| Logistic loss $\ln(1 + e^{-y\langle\theta,x\rangle})$ | yes | $\tfrac14\|x\|_2^2$ |
| $\|x\|_1$ | **no** | not differentiable |
| Hinge loss | **no** | not differentiable |
| ReLU | **no** | gradient discontinuous |

**Closed under sums:** if $f, g$ are $L$-smooth, then $f + g$ is $2L$-smooth (the constants add).

## Why smoothness gives faster optimization

For an $L$-smooth **convex** $f$ with minimizer $x^*$:

- **Gradient descent** with $\eta = 1/L$ gives $f(x_T) - f(x^*) = O(L\|x_0 - x^*\|^2 / T)$ — i.e. $O(1/T)$ convergence in function value, vs. $O(1/\sqrt{T})$ for non-smooth.
- **Mirror descent** with $\mu$-strongly-convex DGF $g$ and $\eta \leq \mu/L$:
$$f(x_T) - f(x^*) \;\leq\; \frac{D_g(x^*, x_0)}{\eta T}.$$
- **Nesterov's accelerated gradient** reaches the optimal $O(L\|x_0 - x^*\|^2 / T^2)$ for smooth-convex functions.

The high-level reason: smoothness lets you take **constant-size steps** (independent of $T$) because the upper quadratic bound guarantees descent. Non-smooth losses force diminishing step sizes $\eta_t \propto 1/\sqrt{t}$.

## Gradient descent lemma (for $L$-smooth functions)

For mirror descent $x_{t+1} = \mathrm{Prox}_g(\eta \nabla f(x_t), x_t)$ with $f$ being $L$-smooth, $g$ being $\mu$-strongly convex, and $0 < \eta \leq \mu/L$:
$$f(x_{t+1}) \leq f(x_t) - \frac{\mu}{2\eta}\|x_t - x_{t+1}\|^2.$$

**Function value is monotonically non-increasing.** Even works *without* convexity of $f$ — smoothness alone gives descent.

## Note: smoothness ≠ Lipschitz function

These are different:
- $f$ is $L$-**Lipschitz** means $|f(x) - f(y)| \leq L\|x - y\|$ — gradient bounded.
- $f$ is $L$-**smooth** means $\nabla f$ is $L$-Lipschitz — Hessian bounded.

Both are common assumptions but not equivalent. Hinge loss is Lipschitz but not smooth; $\|x\|^2$ is smooth but not Lipschitz on $\mathbb{R}^d$ (gradient unbounded).

## See also

- [[Convexity]] — the dual notion of curvature.
- [[Bregman Divergence]] — the smoothness inequality is what bounds $D_f$ from above.
- [[Mirror Descent Analysis]] — uses smoothness to get $O(1/T)$ instead of $O(1/\sqrt{T})$.
- [[Exp-Concavity]] — a stronger curvature condition that also gives fast rates.
