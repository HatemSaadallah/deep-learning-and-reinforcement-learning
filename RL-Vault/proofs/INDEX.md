# Proofs — Exam Prep Index

Bocconi RL 20876 (Celli) — the proofs flagged as "may appear as an open question on the exam".

## Master list

| # | Topic | File | Key trick |
|---|---|---|---|
| 1 | OGD regret $O(\sqrt{T})$ | [[OGD Regret Bound]] | Telescope $\|x_t - x^*\|^2$ |
| 2 | Regret decomposition | [[Regret Decomposition Lemma]] | $\sum_a \mathbb{1}\{a_t=a\}=1$ |
| 3 | ETC analysis $\widetilde{O}(T^{2/3})$ | [[ETC Regret Analysis]] | Hoeffding + union bound + optimize $\epsilon, T_0$ |
| 4 | UCB instance-dep + indep | [[UCB Analysis]] | Decompose $\{a_t = a\}$ into 3 events; split by gap size $\epsilon$ |
| 5 | FTRL stability + regret | [[FTRL Stability and Regret]] | Be-The-Leader + stability of $x_t \to x_{t+1}$ |
| 6 | Mirror descent + simplex OMD | [[Mirror Descent Analysis]] | Three-point Bregman identity; OMD on simplex = Hedge |
| 7 | Lipschitz contextual bandits | [[Lipschitz Contextual Bandits Proof]] | $\epsilon$-net + Cauchy-Schwarz + balance |
| 8 | Regret ↔ Nash in 0-sum games | [[Regret and Equilibria]] | Average no-regret play; minimax |
| 9 | Markov policies suffice | [[Markov Policies Suffice]] | Backward induction via Bellman |
| 10 | VI in generative setting | [[VI Generative Setting]] | Hoeffding on $r$ and $\langle P, V \rangle$, simulation lemma |
| 11 | Policy gradient theorem | [[Policy Gradient Theorem]] | Log-derivative trick + causality |
| 12 | TRPO surrogate objective | [[TRPO Surrogate Objective]] | Performance difference lemma + IS + KL trust region |

## How to use this for exam prep

1. **First pass:** read the proof of every file in order. Each has a "Statement" → "Proof" → "What to remember" structure.
2. **Second pass:** for each one, hide the proof and try to reproduce just from the "Statement" and "Setup" boxes.
3. **Failure mode:** if you forget a step, the proof has explicit "**why this step:**" annotations on the load-bearing moves. Most exam graders care about (a) you used the right tool (e.g. Hoeffding vs. Pinsker), and (b) you got the rate right.
4. **Pitch perfect:** UCB analysis (#4) and TRPO derivation (#12) have the most moving parts. Schedule extra time for those.

## Notation conventions (matches Celli's slides)

- $K$: number of arms
- $T$: horizon
- $\mu_a$: true mean of arm $a$; $\mu^* = \max_a \mu_a$
- $\Delta_a = \mu^* - \mu_a$: [[Sub-optimality Gap]]
- $N_a(t)$: pulls of arm $a$ by time $t$
- $\hat\mu_t(a)$: empirical mean
- $R^T$: pseudo-regret
- $\eta$: step size / learning rate
- $D_R(x, y) = R(x) - R(y) - \langle \nabla R(y), x-y \rangle$: Bregman divergence
- $\gamma \in [0,1)$: discount factor
- $d^\pi_\gamma$: discounted state visitation under policy $\pi$
- $V^\pi, Q^\pi, A^\pi$: state-value, action-value, advantage under $\pi$
