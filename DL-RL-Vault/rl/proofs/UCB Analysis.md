# Proof — UCB Regret (Instance-Dependent and Instance-Independent)

**Tags:** #proof #bandits #ucb
**Topic:** Regret guarantees of UCB1.

## Statement

For $K$-armed [[Stochastic Bandits|stochastic bandits]] with rewards in $[0,1]$, **UCB1** (Auer, Cesa-Bianchi, Fischer 2002) chooses at round $t$:
$$a_t = \arg\max_a \;\hat\mu_a(t-1) + \sqrt{\frac{2 \log t}{N_a(t-1)}}.$$
Then for any sub-optimal arm $a$:
$$\mathbb{E}[N_a(T)] \;\leq\; \frac{8 \log T}{\Delta_a^2} + 1 + \frac{\pi^2}{3}.$$

Consequently:
$$\boxed{\;R^T \;\leq\; \sum_{a:\,\Delta_a > 0} \frac{8 \log T}{\Delta_a} + O(K) \;=\; O\!\left(\frac{K \log T}{\Delta_{\min}}\right) \quad\text{(instance-dependent).}\;}$$
$$\boxed{\;R^T \;\leq\; O\bigl(\sqrt{K T \log T}\bigr) \;=\; \widetilde{O}(\sqrt{KT}) \quad\text{(instance-independent).}\;}$$

The instance-independent bound **matches the [[Bandit Lower Bound]]** up to log factors.

## Setup

Define the confidence radius $\epsilon_t(n) = \sqrt{2 \log T / n}$ for $n$ samples (using $\log T$ form for clean bounds; the $\log t$ variant is similar).

UCB index: $\text{UCB}_a(t) = \hat\mu_a(t-1) + \epsilon_t(N_a(t-1))$.

## Instance-dependent proof

**Step 1 — Decompose the event $\{a_t = a\}$ for $a$ suboptimal.**

If UCB pulls suboptimal arm $a$ at round $t$, then $\text{UCB}_a(t) \geq \text{UCB}_*(t)$. So **at least one of three events holds**:

- $(A_t)$: $\hat\mu_a(t-1) > \mu_a + \epsilon_t(N_a(t-1))$ — $a$'s empirical mean is **too high**.
- $(B_t)$: $\hat\mu_*(t-1) < \mu_* - \epsilon_t(N_*(t-1))$ — optimal's empirical mean is **too low**.
- $(C_t)$: $\mu_a + 2\epsilon_t(N_a(t-1)) > \mu_*$, equivalently $\epsilon_t(N_a(t-1)) > \Delta_a / 2$, equivalently $N_a(t-1) < 8 \log T / \Delta_a^2$.

**Why this trichotomy:** if all three fail,
$$\text{UCB}_a(t) = \hat\mu_a + \epsilon_a \;\overset{\neg A}{\leq}\; \mu_a + 2\epsilon_a \;\overset{\neg C}{\leq}\; \mu_* \;\overset{\neg B}{\leq}\; \hat\mu_* + \epsilon_* = \text{UCB}_*(t),$$
contradicting $\text{UCB}_a(t) \geq \text{UCB}_*(t)$.

So $\{a_t = a\} \subseteq A_t \cup B_t \cup C_t$.

**Step 2 — Bound expected counts via union.**

$$\mathbb{E}[N_a(T)] = \sum_{t=1}^T \Pr[a_t = a] \;\leq\; \sum_t \Pr[A_t] + \sum_t \Pr[B_t] + \sum_t \Pr[C_t].$$

**Step 3 — Bound the count term $C$.**

$N_a(t-1) < 8\log T / \Delta_a^2$ can hold for **at most $\lceil 8 \log T / \Delta_a^2 \rceil$ rounds** (since each pull of $a$ increments the count). So:
$$\sum_t \mathbb{1}[C_t \text{ and } a_t = a] \leq \frac{8 \log T}{\Delta_a^2} + 1.$$

**Step 4 — Bound the concentration terms $A$ and $B$.**

For each $t$ and each possible value of $N_a(t-1) = n$:
$$\Pr\!\left[\hat\mu_a > \mu_a + \sqrt{\tfrac{2 \log T}{n}}\right] \overset{\text{Hoeffding}}{\leq} e^{-2n \cdot 2\log T / n} = e^{-4\log T} = T^{-4}.$$

Summing over $t \leq T$ and $n \leq t$:
$$\sum_t \Pr[A_t] \leq \sum_{t=1}^T \sum_{n=1}^t T^{-4} \leq T^2 \cdot T^{-4} = T^{-2}.$$

(A more careful peeling argument with the $\log t$ variant gives $\sum_t \Pr[A_t] \leq \sum_{n \geq 1} n \cdot 1/n^4 \sim \pi^2/6$. The constant $\pi^2/3$ in the final bound is $\Pr[A] + \Pr[B]$ contributions.)

Same for $B$.

**Step 5 — Combine.**

$$\mathbb{E}[N_a(T)] \;\leq\; \frac{8 \log T}{\Delta_a^2} + 1 + \frac{\pi^2}{3}.$$

By [[Regret Decomposition Lemma]]:
$$R^T = \sum_a \Delta_a \mathbb{E}[N_a(T)] \;\leq\; \sum_{a: \Delta_a > 0} \frac{8 \log T}{\Delta_a} + \Delta_a\!\left(1 + \tfrac{\pi^2}{3}\right) \;=\; O\!\left(\frac{K \log T}{\Delta_{\min}}\right). \qquad \square$$

## Instance-independent proof

**Idea:** split arms by gap size $\epsilon$ to be chosen.

For arms with **small gap** $\Delta_a \leq \epsilon$: contribution at most $\Delta_a \cdot T \leq \epsilon T$ each (since $\mathbb{E}[N_a(T)] \leq T$ trivially); summed over all arms still $\leq \epsilon T$.

$$R^T_{\text{small gap}} \leq \epsilon T.$$

For arms with **large gap** $\Delta_a > \epsilon$: use the instance-dependent bound:
$$\Delta_a \mathbb{E}[N_a(T)] \leq \frac{8 \log T}{\Delta_a} + O(1) \leq \frac{8 \log T}{\epsilon} + O(1).$$
Summed over at most $K$ arms:
$$R^T_{\text{large gap}} \leq \frac{8 K \log T}{\epsilon} + O(K).$$

**Combine:**
$$R^T \leq \epsilon T + \frac{8 K \log T}{\epsilon} + O(K).$$

**Optimize $\epsilon$:**
$$\frac{d}{d\epsilon}\Bigl[\epsilon T + 8K\log T / \epsilon\Bigr] = T - 8K\log T / \epsilon^2 = 0 \;\Longrightarrow\; \epsilon^* = \sqrt{\frac{8 K \log T}{T}}.$$

Substitute:
$$R^T \leq 2 \sqrt{8 K T \log T} + O(K) = O\!\left(\sqrt{KT \log T}\right). \qquad \square$$

## Intuition / what to remember

- **Optimism in the face of uncertainty.** UCB pulls the arm with the highest *plausible* mean. If you're wrong about the optimal arm, you'll learn fast.
- **The $\sqrt{2 \log T / N}$ radius is exactly what Hoeffding requires** for the bad-mean events to sum to a constant — it's "the right confidence width."
- **Three-event decomposition is the key trick** for the instance-dependent bound. Two of the events have constant total probability mass; the third bounds $N_a$ directly.
- **The split by $\epsilon$ converts instance-dep → instance-indep.** Same trick generalizes throughout bandit theory.
- **Tight up to log factors:** $\sqrt{KT}$ matches the [[Bandit Lower Bound]].

## Lecture-style proof (lecture slides 8–13) — via the [[Clean Event]]

Equivalent rate, cleaner structure. The proof the lecture expects on the exam.

**Setup.** Define
$$\mathcal{G} \;=\; \Bigl\{\, |\mu(a) - \hat\mu_t(a)| \leq \sqrt{\tfrac{2 \log T}{N_t(a)}} \quad \forall a, t \,\Bigr\}.$$
By the [[Clean Event]] lemma, $\Pr[\mathcal{G}] \geq 1 - 1/T$.

**Lemma (#pulls of suboptimal arm, on $\mathcal{G}$):**
$$N_{T+1}(a) \;\leq\; \frac{8 \log T}{\Delta_a^2} + 1.$$

*Proof.* Let $t$ be the **last** round on which arm $a$ is played. Since $a_t = a$:
$$\mathrm{UCB}_{t-1}(a) \;\geq\; \mathrm{UCB}_{t-1}(a^*).$$

On $\mathcal{G}$: $\hat\mu_{t-1}(a^*) \geq \mu^* - \sqrt{2 \log T / N_{t-1}(a^*)}$, so
$$\mathrm{UCB}_{t-1}(a^*) = \hat\mu_{t-1}(a^*) + \sqrt{\tfrac{2 \log T}{N_{t-1}(a^*)}} \;\geq\; \mu^*.$$

On $\mathcal{G}$: $\hat\mu_{t-1}(a) \leq \mu(a) + \sqrt{2 \log T / N_{t-1}(a)}$, so
$$\mathrm{UCB}_{t-1}(a) = \hat\mu_{t-1}(a) + \sqrt{\tfrac{2 \log T}{N_{t-1}(a)}} \;\leq\; \mu(a) + 2\sqrt{\tfrac{2 \log T}{N_{t-1}(a)}}.$$

Chaining: $\mu^* \leq \mu(a) + 2\sqrt{2 \log T / N_{t-1}(a)}$, i.e.
$$\Delta_a \leq 2\sqrt{\tfrac{2 \log T}{N_{t-1}(a)}} \;\Longrightarrow\; N_{t-1}(a) \leq \frac{8 \log T}{\Delta_a^2}.$$

Since $t$ is the last play, $N_{T+1}(a) = N_t(a) = N_{t-1}(a) + 1 \leq 8 \log T / \Delta_a^2 + 1$. $\square$

**Instance-dependent bound.** Split by whether $\mathcal{G}$ holds:

- **On $\mathcal{G}$** (prob $\geq 1 - 1/T$): by [[Regret Decomposition Lemma]] + lemma above,
$$R^T \;\leq\; \sum_a \Delta_a \cdot N_{T+1}(a) \;\leq\; 6 \log T \sum_a \frac{1}{\Delta_a}.$$
- **On $\mathcal{G}^c$** (prob $\leq 1/T$): trivially $R^T \leq T$.

Combine:
$$\mathbb{E}[R^T] \leq \frac{1}{T} \cdot T + \Bigl(1 - \frac{1}{T}\Bigr) \cdot 6 \log T \sum_a \frac{1}{\Delta_a} \;\leq\; 1 + 6 \log T \sum_{a : \Delta_a > 0} \frac{1}{\Delta_a} \;=\; O\!\left(\log T \sum_a \tfrac{1}{\Delta_a}\right). \quad \square$$

**Instance-independent bound.** Pick threshold $\beta = \sqrt{K \log T / T}$. Split arms:

- **Arms with $\Delta_a \leq \beta$** (call them "near-optimal"): on $\mathcal{G}$, total regret from them is
$$\sum_{a:\Delta_a \leq \beta} \Delta_a N_{T+1}(a) \leq \beta \sum_a N_{T+1}(a) = \beta T = \sqrt{K T \log T}.$$
- **Arms with $\Delta_a > \beta$**: each contributes
$$\Delta_a \cdot N_{T+1}(a) \leq \Delta_a \cdot \frac{8 \log T}{\Delta_a^2} = \frac{8 \log T}{\Delta_a} < \frac{8 \log T}{\beta} = 8 \sqrt{T \log T / K}.$$
At most $K$ such arms: total $\leq 8K \sqrt{T \log T / K} = 8\sqrt{KT \log T}$.

Combined: $R^T = O(\sqrt{KT \log T})$. $\square$

**Why this proof is more memorable than the textbook three-event one above:**
- A single high-probability event $\mathcal{G}$ replaces three per-round events.
- The bound on $N_{T+1}(a)$ is **deterministic on $\mathcal{G}$**, not in expectation.
- The threshold $\beta = \sqrt{K \log T / T}$ for the instance-independent bound is chosen *upfront* from the problem geometry rather than optimized after the fact.

## See also

- [[Clean Event]] — the lemma that powers the lecture-style proof.
- [[Regret Decomposition Lemma]] — converts $\mathbb{E}[N_a]$ to regret.
- [[Hoeffding Inequality]] — the concentration tool.
- [[Bandit Lower Bound]] — the $\Omega(\sqrt{KT})$ that this matches.
- [[Explore-Then-Commit]] — strictly worse by $T^{1/6}$ in the exponent of $T$.
- [[UCB1]] — the algorithm note.
- [[Optimism Principle]] — the general design principle.
