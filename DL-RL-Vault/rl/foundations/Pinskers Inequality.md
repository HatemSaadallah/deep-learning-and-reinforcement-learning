# Pinsker's Inequality

**Tags:** #information-theory #lemma

Connects [[KL Divergence]] to total variation distance.

## Statement

For distributions $p, q$ on $\Omega$ and any event $A \subseteq \Omega$:
$$2(p(A) - q(A))^2 \leq \mathrm{KL}(p, q).$$

Equivalently, $\|p - q\|_{TV} \leq \sqrt{\mathrm{KL}(p,q)/2}$.

## Why it matters

It is the standard tool for arguing **"if $p$ and $q$ are close in KL, no statistical test can distinguish them well"**, which is the heart of every minimax lower bound — including the [[Bandit Lower Bound]].

## Use in the random-coins lemma

For product distributions $p = p_1 \times \cdots \times p_T$ over $\{0,1\}^T$:
$$|p(A) - q(A)| \leq \sqrt{\mathrm{KL}(p,q)/2} \;\overset{\text{chain}}{=}\; \sqrt{T \cdot \mathrm{KL}(p_1, q_1)/2} \leq \epsilon\sqrt{T}.$$

So after $T$ rounds, a biased and a fair coin can be distinguished only if $T \gtrsim 1/\epsilon^2$. This is exactly why we need $T = \Omega(K/\epsilon^2)$ in [[Best-Arm Identification]].
