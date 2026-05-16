# Stage 1: Stochastic bandits

Reproduces the regret behaviors from the Bocconi RL Part 2 lecture:
- **Uniform / FTL** → linear regret
- **ETC** → $\widetilde{O}(T^{2/3})$
- **UCB1 / Thompson Sampling** → $\widetilde{O}(\sqrt{KT})$, matching the [[Bandit Lower Bound]]

## Run

```bash
source ../../.venv/bin/activate
python plot_regret.py
```

Outputs `regret_curves.png` (linear) and `regret_loglog.png` (log-log, to read off the exponent in $T$).

## Files

- `env.py` — `BernoulliBandit`, `GaussianBandit`
- `agents.py` — `FollowTheLeader`, `UniformExploration`, `EpsilonGreedy`, `ExploreThenCommit`, `UCB1`, `ThompsonSamplingBeta`
- `experiment.py` — episode runner, multi-seed averaging
- `plot_regret.py` — main entry point

## Notation (matches lecture)

| Code | Slides |
|---|---|
| `env.gaps[a]` | $\Delta_a = \mu^* - \mu_a$ |
| `agent.N[a]` | $N_a(t)$ |
| `agent.mu_hat[a]` | $\hat\mu_t(a)$ |
| `regret[t]` | $R^t = t\mu^* - \sum_{s \leq t} \mu_{a_s}$ |

## Next steps

- Add KL-UCB (uses [[KL Divergence]] for tighter Bernoulli bound)
- Add adversarial bandits (Exp3) — uses [[Hedge - Multiplicative Weights]]
- Add gap-dependent ETC (Lattimore & Szepesvári Ch. 6)
