# RLHF and DPO (RL for LLMs)

**Tags:** #mdp #algorithm #deep-rl #llm
**Source:** Christiano et al. 2017 (RLHF), Ouyang et al. 2022 (InstructGPT), Rafailov et al. 2023 (DPO). Celli "RL & LLMs" lecture, 2026-05-13.

The dominant paradigm for **aligning large language models** with human preferences. RL provides the optimization engine when the objective ("be helpful, honest, harmless") can't be expressed as a differentiable loss.

## The LLM-as-MDP framing

| MDP component | LLM correspondent |
|---|---|
| State $s_t$ | prompt + tokens generated so far $(x, y_{1:t})$ |
| Action $a_t$ | next token $y_{t+1}$ |
| Policy $\pi_\theta(\cdot \mid s_t)$ | LM output distribution |
| Transition | **deterministic concatenation** $(s_t, a_t) \to (s_t, a_t)$ |
| Reward | (typically) **terminal only**: $r_\psi(x, y)$ when `<EOS>` reached |
| Discount $\gamma$ | $1$ |

The state is full history (autoregressive → non-Markov) but **made Markov by the construction** (treating the prefix as the state). The whole episode = one completion.

## The three-stage LLM pipeline

```
Pre-training → Supervised Fine-Tuning (SFT) → RLHF Alignment
              ↑ next-token prediction          ↑ human values
```

Phases I + II are imitation learning. They **don't steer the model away from bad outputs**, only toward good ones. RLHF fixes this by providing negative-signal feedback (a worse completion is *preferred against* a better one).

## RLHF pipeline (InstructGPT, Ouyang et al. 2022)

### Three components

1. $\pi_\theta$ — instruction-fine-tuned LM (starting point).
2. $r_\psi$ — **reward model**, a neural net mapping $(x, y) \to \mathbb{R}$. Initialized from a pretrained LM.
3. $V_\phi$ — value network (critic in PPO). Initialized from the reward model.

### Stage 1: Train the reward model

Collect human preference data: for each prompt $x$, generate $K$ completions $y_1, \dots, y_K$, ask humans to rank them.

**Bradley-Terry preference model** assumes $\Pr[y_i \succ y_j \mid x] = \sigma(r(x, y_i) - r(x, y_j))$ where $\sigma$ is sigmoid. Fit by:
$$\mathcal{L}(\psi) \;=\; -\mathbb{E}_{(x, y_{\text{win}}, y_{\text{lose}})}\!\Bigl[\log \sigma\bigl(r_\psi(x, y_{\text{win}}) - r_\psi(x, y_{\text{lose}})\bigr)\Bigr].$$
Standard supervised classification loss on pairs.

### Stage 2: Policy optimization with PPO

Run [[PPO and GRPO|PPO]] using $r_\psi$ as the reward:
- $\hat A_t = r_\psi(x, y_{1:T+1}) - V_\phi(x, y_{1:t})$ ← advantage at token $t$ (in practice use GAE).
- Update $\pi_\theta$ to maximize the clipped surrogate.

### Naive baseline: Best-of-N sampling

Skip RL entirely:
1. Generate $N$ completions.
2. Return the one with highest $r_\psi$.

**Pros:** simple, no RL training.
**Cons:** needs $N$ generations at inference time. Wasteful.

## The over-optimization problem

Running PPO blindly causes two pathologies:

1. **Reward hacking** — $\pi_\theta$ learns to exploit imperfections in $r_\psi$ (e.g. produces outputs that score high but are degenerate/manipulative).
2. **Distribution shift** — $r_\psi$ was trained on completions from $\pi^{\mathrm{FT}}$. If $\pi_\theta$ drifts too far from $\pi^{\mathrm{FT}}$, $r_\psi$'s predictions become unreliable.
3. **Capability loss** — fine-tuning too much can "break" the LM's general abilities.

### Solution: KL penalty against the SFT model

Add a regularizer keeping $\pi_\theta$ close to $\pi^{\mathrm{FT}}$:
$$J(\theta) \;=\; \mathbb{E}_{(x, y) \sim \mathcal{D}}\!\left[r_\psi(x, y) - \beta\,\log\frac{\pi_\theta(y|x)}{\pi^{\mathrm{FT}}(y|x)}\right] \;+\; \eta\,\mathbb{E}_{x \sim \mathcal{D}_{\mathrm{pretrain}}}\!\bigl[\log \pi_\theta(x)\bigr].$$

- $\beta\log(\pi_\theta / \pi^{\mathrm{FT}})$ → KL penalty enforcing $\pi_\theta \approx \pi^{\mathrm{FT}}$.
- $\eta \log\pi_\theta(x)$ → pretraining loss preserving capabilities.

**Implementation trick:** absorb the KL into per-token rewards
$$r_t = -\beta\log\frac{\pi_\theta(y_{t+1}|x, y_{1:t})}{\pi^{\mathrm{FT}}(y_{t+1}|x, y_{1:t})}, \quad r_T = r_\psi(x, y) - \beta\log\frac{\pi_\theta(y_T|\cdot)}{\pi^{\mathrm{FT}}(y_T|\cdot)},$$
then run vanilla PPO on this augmented reward.

## Closed-form solution of the RLHF problem

Ignoring the pretraining term, the RLHF objective is
$$\max_\theta \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta}\!\left[r(x, y) - \beta\log\frac{\pi_\theta(y|x)}{\pi^{\mathrm{FT}}(y|x)}\right].$$

Without the neural-network parameterization, this is equivalent to:
$$\min_\pi \mathbb{E}\!\left[\mathrm{KL}\bigl(\pi(\cdot|x) \,\|\, \pi_r(\cdot|x)\bigr) - \log Z(x)\right],$$
where
$$\pi_r(y|x) \;=\; \frac{\pi^{\mathrm{FT}}(y|x)\, \exp\bigl(r(x, y)/\beta\bigr)}{Z(x)}, \quad Z(x) = \sum_y \pi^{\mathrm{FT}}(y|x)\, e^{r(x, y)/\beta}.$$

The KL is minimized at $\pi^* = \pi_r$ — a closed form! Inverting:
$$r(x, y) \;=\; \beta\log\frac{\pi(y|x)}{\pi^{\mathrm{FT}}(y|x)} + \beta\log Z(x).$$

**Reward $\leftrightarrow$ optimal policy is a bijection** (up to the constant $\beta\log Z(x)$ which doesn't affect preference comparisons).

## Direct Preference Optimization (DPO)

Rafailov et al. 2023. **Exploits the closed form** to skip training a reward model entirely.

### PPO-style approach (3 stages)
1. Collect human preferences.
2. Learn $r_\psi$ to fit preferences.
3. Learn $\pi$ via PPO on $r_\psi$ with KL penalty.

### DPO approach (1 stage)
1. Collect human preferences.
2. Learn $\pi$ directly **such that $r_\pi$ fits preferences**.

The substitution: plug $r_\pi(x, y) = \beta\log(\pi(y|x) / \pi^{\mathrm{FT}}(y|x))$ (dropping $Z(x)$ — cancels in pairwise) into the Bradley-Terry loss:

$$\boxed{\;\min_\theta \sum_{(x, y_i, y_j) \in \mathcal{D},\, y_i \succ y_j} -\log \sigma\!\left[\beta\log\frac{\pi_\theta(y_i|x)}{\pi^{\mathrm{FT}}(y_i|x)} - \beta\log\frac{\pi_\theta(y_j|x)}{\pi^{\mathrm{FT}}(y_j|x)}\right].\;}$$

**Pure supervised loss.** No reward model, no critic, no rollouts, no RL.

### Pros and cons

| | PPO-based RLHF | DPO |
|---|---|---|
| Pipeline complexity | reward model + value model + PPO | single supervised loss |
| Training stability | needs careful hyperparameter tuning | standard SGD |
| On-policy data | needs continuous generation | only the static preference dataset |
| Sample efficiency | low (PPO is on-policy) | high |
| Performance | strong (when tuned) | competitive; **debated** (see Xu et al. ICML 2024) |

## See also

- [[PPO and GRPO]] — the RL optimizer used in stage 2 of classic RLHF.
- [[TRPO Surrogate Objective]] — PPO's theoretical predecessor.
- [[KL Divergence]] — the regularizer that constrains drift from $\pi^{\mathrm{FT}}$.
- [[Markov Decision Process]] — what the LLM-as-MDP framing leverages.

## References

- Christiano et al., *Deep reinforcement learning from human preferences*, NeurIPS 2017.
- Ouyang et al., *Training language models to follow instructions with human feedback*, NeurIPS 2022 (InstructGPT).
- Rafailov et al., *Direct preference optimization: Your language model is secretly a reward model*, NeurIPS 2023.
- Shao et al., *DeepSeekMath: Pushing the limits of mathematical reasoning in open language models*, 2024 (GRPO).
