# Proof — Value Iteration in the Generative Setting

**Tags:** #proof #mdp #sample-complexity
**Topic:** Sample complexity of value iteration using a generative model.

## Statement

Consider an infinite-horizon discounted MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ with rewards in $[0,1]$. Assume access to a **generative model**: given any $(s, a)$, can sample $r \sim R(s,a)$ and $s' \sim P(\cdot \mid s, a)$.

**Algorithm:** sample $N$ transitions per $(s, a)$. Form empirical model $\hat P, \hat R$. Run value iteration on the empirical MDP $\hat{\mathcal{M}}$ to obtain $\hat V^*$.

**Theorem:** with
$$N \;=\; \widetilde{O}\!\left(\frac{1}{(1-\gamma)^4 \,\varepsilon^2}\right) \quad \text{per } (s,a),$$
i.e. total samples $|\mathcal{S}||\mathcal{A}| \cdot N$, we get
$$\boxed{\;\|\hat V^* - V^*\|_\infty \leq \varepsilon \;\;\text{with probability } \geq 1-\delta.\;}$$

## Setup

- $V_{\max} := 1/(1-\gamma)$ — uniform upper bound on values for rewards in $[0,1]$.
- Bellman operator $\mathcal{T} V(s) := \max_a \bigl[R(s,a) + \gamma \langle P(\cdot|s,a), V\rangle\bigr]$. $\mathcal{T}$ is a $\gamma$-contraction in $\|\cdot\|_\infty$.
- Empirical Bellman operator $\hat{\mathcal{T}} V(s) := \max_a \bigl[\hat R(s,a) + \gamma\langle \hat P(\cdot|s,a), V\rangle\bigr]$.
- $V^* = \mathcal{T} V^*$ (unique fixed point); $\hat V^* = \hat{\mathcal{T}} \hat V^*$.

## Proof

**Step 1 — Bellman-residual decomposition.**

Both $V^*$ and $\hat V^*$ are fixed points. By the contraction property:
$$\|\hat V^* - V^*\|_\infty \;=\; \|\hat{\mathcal{T}}\hat V^* - \mathcal{T} V^*\|_\infty \;\leq\; \|\hat{\mathcal{T}}\hat V^* - \hat{\mathcal{T}} V^*\|_\infty + \|\hat{\mathcal{T}} V^* - \mathcal{T} V^*\|_\infty.$$
The first term is $\leq \gamma\|\hat V^* - V^*\|_\infty$ ($\hat{\mathcal{T}}$ is a $\gamma$-contraction). Rearranging:
$$(1-\gamma) \|\hat V^* - V^*\|_\infty \;\leq\; \|\hat{\mathcal{T}} V^* - \mathcal{T} V^*\|_\infty.$$
$$\|\hat V^* - V^*\|_\infty \;\leq\; \frac{1}{1-\gamma}\,\|\hat{\mathcal{T}} V^* - \mathcal{T} V^*\|_\infty. \tag{$\star$}$$

**This is the key reduction:** bounding the error on the *true* $V^*$ (a fixed function, not random) controls the propagated error.

**Step 2 — Bound the Bellman residual on $V^*$.**

For each $(s,a)$:
$$|\hat{\mathcal{T}} V^*(s) - \mathcal{T} V^*(s)| \;\leq\; \max_a\Bigl[|\hat R(s,a) - R(s,a)| + \gamma\,|\langle \hat P - P,\, V^*\rangle|\Bigr].$$

(Pulling absolute values inside max.)

Both terms are concentration of sample averages around a population mean, with $V^*$ a *fixed* function (not depending on the data — load-bearing!).

**Sub-step 2a — Reward concentration (Hoeffding).** Rewards in $[0,1]$, $N$ i.i.d. samples:
$$\Pr\!\left[|\hat R(s,a) - R(s,a)| > t\right] \leq 2 e^{-2Nt^2}.$$
For $t = \sqrt{\log(2/\delta')/(2N)}$, this is $\leq \delta'$.

**Sub-step 2b — Transition concentration (Hoeffding on $\langle P, V^*\rangle$).**

Since $\langle \hat P, V^*\rangle$ is the average of $N$ i.i.d. samples of $V^*(s')$ with $V^*(s') \in [0, V_{\max}]$:
$$\Pr\!\left[|\langle \hat P - P, V^*\rangle| > t\right] \leq 2 e^{-2Nt^2/V_{\max}^2}.$$
For $t = V_{\max} \sqrt{\log(2/\delta')/(2N)}$: probability $\leq \delta'$.

**Step 3 — Union bound over $(s,a)$.**

There are $2|\mathcal{S}||\mathcal{A}|$ events (reward and transition per pair). Set $\delta' = \delta / (2|\mathcal{S}||\mathcal{A}|)$. All concentration events hold simultaneously w.p. $\geq 1 - \delta$.

Under this good event:
$$\|\hat{\mathcal{T}} V^* - \mathcal{T} V^*\|_\infty \;\leq\; \sqrt{\frac{\log(4|\mathcal{S}||\mathcal{A}|/\delta)}{2N}} + \gamma\,V_{\max}\sqrt{\frac{\log(4|\mathcal{S}||\mathcal{A}|/\delta)}{2N}} \;\leq\; \frac{2V_{\max}}{(1-\gamma)} \cdot c\sqrt{\frac{\log(\cdot)}{N}}.$$

Hmm let me redo the bound carefully:
$$\|\hat{\mathcal{T}} V^* - \mathcal{T} V^*\|_\infty \leq (1 + \gamma V_{\max}) \sqrt{\frac{\log(4|\mathcal{S}||\mathcal{A}|/\delta)}{2N}} \leq \frac{2}{1-\gamma}\sqrt{\frac{\log(4|\mathcal{S}||\mathcal{A}|/\delta)}{2N}}.$$

**Step 4 — Combine with $(\star)$.**

$$\|\hat V^* - V^*\|_\infty \;\leq\; \frac{1}{1-\gamma} \cdot \frac{2}{1-\gamma} \sqrt{\frac{\log(\cdot)}{2N}} \;=\; \frac{2}{(1-\gamma)^2}\sqrt{\frac{\log(\cdot)}{2N}}.$$

**Step 5 — Set $N$ to achieve $\varepsilon$.**

Set the RHS $\leq \varepsilon$ and solve for $N$:
$$N \;\geq\; \frac{2 \log(4|\mathcal{S}||\mathcal{A}|/\delta)}{(1-\gamma)^4\, \varepsilon^2} \;=\; \widetilde{O}\!\left(\frac{1}{(1-\gamma)^4\,\varepsilon^2}\right) \text{ per }(s,a). \qquad \square$$

## Sharper bounds and what they cost

The $(1-\gamma)^{-4}$ is **not tight**. Azar et al. (2013) achieve $\widetilde{O}(1/((1-\gamma)^3 \varepsilon^2))$ using a Bernstein-style argument that exploits the *variance* of $V^*$ (which is small in many MDPs). Match the lower bound.

For exam purposes the Hoeffding-based bound above is the standard derivation.

## Intuition / what to remember

- **The fixed-point reduction $(\star)$ is the heart of the proof.** It converts the recursive question ("how does empirical error propagate through value iteration?") into a one-shot concentration question on the residual at $V^*$.
- **$V^*$ being deterministic (not data-dependent) is crucial.** This is why one bounds the residual on $V^*$, not on $\hat V^*$. Bounding on $\hat V^*$ would require a uniform concentration over a class of functions, much harder.
- **The $1/(1-\gamma)$ factors:** one from contraction ($\star$), one from $V_{\max} = 1/(1-\gamma)$ in transitions. Squared gives $1/(1-\gamma)^4$.
- **Generative model is a strong assumption** — you can sample any $(s,a)$ at will. In standard online RL you can only sample reachable ones.

## See also

- [[Markov Policies Suffice]] — justifies value iteration in the first place.
- [[Hoeffding Inequality]] — the concentration tool.
- [[Q-learning]] — sample-based version when generative model unavailable.
