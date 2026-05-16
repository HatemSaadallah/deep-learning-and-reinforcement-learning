# Clean Event

**Tags:** #foundational #lemma #concentration #bandits

The high-probability event that **all empirical means are simultaneously within their confidence radius of the true means**. Load-bearing for Celli's lecture-style proof of UCB.

## Statement

For [[Stochastic Bandits|stochastic bandits]] with rewards in $[0,1]$, define
$$\mathcal{G} \;:=\; \Bigl\{\, |\mu(a) - \hat\mu_t(a)| \;\leq\; \sqrt{\tfrac{2 \log T}{N_t(a)}} \quad \forall a \in [K],\; \forall t \in [T] \,\Bigr\}.$$

**Lemma:**
$$\boxed{\;\Pr[\mathcal{G}] \;\geq\; 1 - \frac{1}{T}.\;}$$

## The trap: you cannot apply [[Hoeffding Inequality|Hoeffding]] directly

Hoeffding gives, for a **fixed** $n$:
$$\Pr\!\left[\bigl|\hat\mu_n - \mu\bigr| > \epsilon\right] \;\leq\; 2 e^{-2n \epsilon^2}.$$

But in UCB analysis, **$N_t(a)$ is itself a random variable** — it depends on which arms the algorithm has chosen, which depends on past rewards. You can't just "plug $N_t(a)$ into Hoeffding" because the bound assumes $n$ is fixed before observing data.

This is the standard pitfall in concentration arguments for adaptive sampling.

## The fix: peeling / union bound over $n$

Pretend each arm $a$ has access to its own i.i.d. reward stream $X_1(a), X_2(a), \dots, X_T(a) \sim \mathcal{D}_a$. The algorithm only "uses" the first $N_t(a)$ of these — so $\hat\mu_t(a) = \tfrac{1}{N_t(a)}\sum_{s \leq N_t(a)} X_s(a)$ equals $\bar X_{N_t(a)}(a)$ where $\bar X_n(a)$ is the empirical mean of the first $n$ pre-generated samples.

Now the bound we want is:
$$|\mu(a) - \bar X_n(a)| \leq \sqrt{\tfrac{2 \log T}{n}} \quad \forall n \in [T].$$

For **each fixed $n$**, Hoeffding gives
$$\Pr\!\left[|\mu(a) - \bar X_n(a)| > \sqrt{\tfrac{2 \log T}{n}}\right] \;\leq\; 2 e^{-2n \cdot 2\log T / n} = \frac{2}{T^4}.$$

**Union bound** over $n \in [T]$ for arm $a$:
$$\Pr\!\left[\exists n : |\mu(a) - \bar X_n(a)| > \sqrt{\tfrac{2 \log T}{n}}\right] \leq \frac{2}{T^3}.$$

**Union bound** over $K$ arms:
$$\Pr[\mathcal{G}^c] \leq \frac{2K}{T^3} \leq \frac{1}{T} \quad (\text{for } T \geq 2K). \qquad \square$$

## How it's used (UCB analysis sketch)

The proof of UCB regret bounds (see [[UCB Analysis]]) splits into two cases:

1. **On $\mathcal{G}$** (probability $\geq 1 - 1/T$): for any suboptimal arm $a$,
$$N_{T+1}(a) \;\leq\; \frac{8 \log T}{\Delta_a^2}.$$
Proof: let $t$ be the last round $a$ was played. By UCB's selection rule, $\mathrm{UCB}_{t-1}(a) \geq \mathrm{UCB}_{t-1}(a^*) \geq \mu^*$ (the last inequality from $\mathcal{G}$). Also $\mathrm{UCB}_{t-1}(a) \leq \mu(a) + 2\sqrt{2 \log T / N_{t-1}(a)}$ on $\mathcal{G}$. Combining: $\Delta_a \leq 2\sqrt{2 \log T / N_{t-1}(a)}$, hence the bound.

2. **On $\mathcal{G}^c$** (probability $\leq 1/T$): trivially $R^T \leq T$.

Total: $\mathbb{E}[R^T] \leq \tfrac{1}{T} \cdot T + (1 - \tfrac{1}{T}) \cdot 6 \log T \sum_a \tfrac{1}{\Delta_a} = O(\log T \sum_a 1/\Delta_a)$.

## Why this proof style is preferred

- **Deterministic conclusion on $\mathcal{G}$.** The bound $N_{T+1}(a) \leq 8 \log T/\Delta_a^2$ holds *for every $\omega$* in the clean event — no expectations needed inside.
- **Modular.** The clean event is reused without modification for KL-UCB, UCB-V, and various contextual / linear UCB variants. Each just changes the confidence radius and the corresponding probability bound.
- **Forces you to confront the random-$N$ issue.** The naive "Hoeffding at $N_t(a)$" mistake is so common that the peeling-style proof is now standard pedagogy in modern bandit textbooks (Slivkins; Lattimore-Szepesvári Ch. 7).

## See also

- [[UCB Analysis]] — the regret proof building on this lemma.
- [[Hoeffding Inequality]] — the per-$n$ concentration, plus the trap above.
- [[Stochastic Bandits]] — the setting.
