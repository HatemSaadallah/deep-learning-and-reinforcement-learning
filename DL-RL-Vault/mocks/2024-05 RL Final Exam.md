# 2024-05-16 — 20876 Final Exam

**Tags:** #mock #exam #practice

- **Date:** 16 May 2024
- **Duration:** 1h30
- **Total points:** 31
- **Questions:** 16 (Q1–Q11 multi-choice, 1pt each; Q12–Q16 open, 3–6pt)

> "For multiple choice questions, you get 0 for a wrong answer. Fill in the bubbles for **all** correct choices: there may be more than one correct choice, but there is always at least one correct choice. Questions with more than one correct answer are marked with **(*)**. No partial credit on multiple answer questions."

---

## Multiple-choice (Q1–Q11, 1pt each)

**Q1** (1pt): What is the name of the method used to differentiate through a stochastic component?
- [ ] Reparameterization trick
- [ ] Monte Carlo estimation
- [ ] Ancestral sampling
- [ ] Variational approximation

*Topic: DL — VAEs / stochastic gradients.*

**Q2** (1pt) **(*)**: Which of the following are true regarding the forward pass in backpropagation?
- [ ] It computes the activations of each layer in the neural network
- [ ] It calculates the derivatives of the loss with respect to the activations
- [ ] It updates the weights and biases of the network
- [ ] It stores the intermediate values for use in the backward pass

*Topic: DL — backpropagation.*

**Q3** (1pt): Which of the following is NOT a property of the Evidence Lower Bound (ELBO)?
- [ ] It is a lower bound on the log-likelihood
- [ ] It is tight when the variational distribution matches the true posterior
- [ ] It can be expressed as the sum of reconstruction error and the distance to the prior
- [ ] It is used to only measure the distance between two probability distributions

*Topic: DL — VAEs / ELBO.*

**Q4** (1pt) **(*)**: Which of the following statements are true regarding the Universal Approximation Theorem?
- [ ] It guarantees that a neural network will always find the optimal approximation of a function
- [ ] It states that a neural network with a single hidden layer can approximate any continuous function to arbitrary precision
- [ ] It is only applicable to neural networks with specific activation functions
- [ ] It applies to both shallow and deep neural networks

*Topic: DL — UAT.*

**Q5** (1pt): What is the primary goal of the generator network in a Generative Adversarial Network (GAN)?
- [ ] To classify input data as real or generated
- [ ] To discriminate between real and generated samples
- [ ] To generate samples that are indistinguishable from real data
- [ ] To minimize the Wasserstein distance between distributions

*Topic: DL — GANs.*

**Q6** (1pt) **(*)**: Which of the following are examples of explicit regularization techniques?
- [ ] Dropout
- [ ] L2 regularization
- [ ] Early stopping
- [ ] L1 regularization

*Topic: DL — regularization.*

**Q7** (1pt): What is the primary goal of the maximum likelihood criterion in training machine learning models?
- [ ] To maximize the probability of the observed data given the model's parameters
- [ ] To minimize the difference between predicted and actual values
- [ ] To find the model parameters that maximize the likelihood of the data
- [ ] To minimize the negative log-likelihood of the data

*Topic: DL — MLE.*

**Q8** (1pt): Which of the following statements about Masked Self-Attention in the Transformer decoder is **FALSE**?
- [ ] It allows for simultaneous training on all tokens in the target sequence
- [ ] It enables each token to attend to all previously generated tokens
- [ ] It is identical to the self-attention mechanism used in the encoder
- [ ] It helps in generating the target sequence one token at a time

*Topic: DL — Transformers.*

**Q9** (1pt): How can we define $Q$-learning?
- [ ] on-policy policy-iteration algorithm
- [ ] off-policy policy-iteration algorithm
- [ ] on-policy value-iteration algorithm
- [ ] off-policy value-iteration algorithm

*Topic: [[Q-learning]] — off-policy, value-based.*

**Q10** (1pt): Consider a stochastic multi-armed bandit problem with $K$ arms, and let $\Delta_{\max} = \max_a \Delta_a$, $\Delta_{\min} = \min_{a:\mu_a<\mu^*} \Delta_a$. Running an $\epsilon$-greedy strategy has expected regret:
- [ ] $R^T \geq \frac{(1-\epsilon)}{K}\Delta_{\max} T$
- [ ] $R^T \geq \frac{(1-\epsilon)}{K}\Delta_{\min} T$
- [ ] $R^T \geq \epsilon \frac{K-1}{K}\Delta_{\max} T$
- [ ] $R^T \geq \epsilon \frac{K-1}{K}\Delta_{\min} T$
- [ ] $R^T \geq \epsilon K \Delta_{\max} T$
- [ ] $R^T \geq \epsilon K \Delta_{\min} T$

*Topic: [[Epsilon-Greedy]] — linear regret lower bound.*

**Q11** (1pt): In the generative setting, VI requires a number of samples to get $\epsilon$-optimality with the following dependence on the number of states $S$:
- [ ] $\widetilde{O}(S)$
- [ ] $\widetilde{O}(S^2)$
- [ ] $\widetilde{O}(S^{-1})$
- [ ] $\widetilde{O}(\log S)$

*Topic: [[VI Generative Setting]] — sample complexity in $S$.*

---

## Open questions (Q12–Q16)

**Q12** (3pts): For a given function $f$ and mirror map $g$, define the Bregman divergence $D_g(x, y)$ and write a mirror descent update on function $f$ to go from $x_t$ to $x_{t+1}$.

*Topic: [[Bregman Divergence]] + [[Mirror Descent Analysis]].*

**Q13** (3pts): Prove that the *non-negative homogeneity property* of the ReLU function holds for $\alpha \in \mathbb{R}^+$:
$$\mathrm{ReLU}[\alpha \cdot z] = \alpha \cdot \mathrm{ReLU}[z].$$

*Topic: DL — ReLU properties (case split on sign of $z$).*

**Q14** (4pts): How do the gradients of the loss function change when L2 regularization is added?
$$\hat\phi = \arg\min_\phi \Bigl[\sum_{i=1}^I \ell_i[\boldsymbol{x}_i, \boldsymbol{y}_i] + \lambda \sum_j \phi_j^2\Bigr]$$

*Topic: DL — L2/Tikhonov regularization, weight decay.*

**Q15** (4pts): Provide the statement of the $Q$-version of the policy gradient theorem and prove it.

*Topic: [[Policy Gradient Theorem]] — full proof needed.*

**Q16** (6pts): In which setting can we apply the EXP3 algorithm? Describe the algorithm and its regret guarantees.

*Topic: [[EXP3]] + [[Adversarial Bandits]] — algorithm + $O(\sqrt{KT \log K})$ regret.*

---

## Answer key

> ⚠️ Try the questions first. The key is intentionally placed last.

<details>
<summary>Click to reveal answers</summary>

**Q1:** Reparameterization trick. (The classic "differentiate through a $\mathcal{N}(0, I)$ sample" used in VAEs.)

**Q2 (*):** ✅ "computes the activations" + ✅ "stores the intermediate values for use in the backward pass". The other two describe the backward pass / weight update.

**Q3:** "It is used to only measure the distance between two probability distributions." (False — ELBO is a lower bound on log-likelihood, not just a distance.)

**Q4 (*):** ✅ "single hidden layer can approximate any continuous function" + ✅ "applies to both shallow and deep". The first option is wrong (UAT is existence, not optimization).

**Q5:** "To generate samples that are indistinguishable from real data."

**Q6 (*):** ✅ L2 regularization + ✅ L1 regularization. Dropout and early stopping are *implicit* regularization.

**Q7:** All three of "maximize prob of observed data", "find params that maximize likelihood", and "minimize NLL" describe MLE. The single-answer phrasing is ambiguous; pick the most direct: **"To find the model parameters that maximize the likelihood of the data"**.

**Q8:** "It is identical to the self-attention mechanism used in the encoder." (False — encoder is bidirectional; decoder uses causal masking.)

**Q9:** **off-policy value-iteration algorithm.** See [[Q-learning]] — off-policy because of $\max_{a'}$ in the target, value-iteration because it iterates the Bellman optimality operator (not a policy improvement step). 

**Q10:** $R^T \geq \epsilon \frac{K-1}{K}\Delta_{\min} T$. See [[Epsilon-Greedy]] — fraction $\epsilon$ of rounds we explore uniformly, each suboptimal arm pulled with prob $1/K$, each contributes at least $\Delta_{\min}$.

**Q11:** $\widetilde{O}(S)$ (Bernstein-based: optimal rate is $\widetilde{O}(SA/((1-\gamma)^3 \epsilon^2))$ — **linear** in $S$, not $S^2$). See [[VI Generative Setting]]. The trick is bounding the Bellman residual at $V^*$ instead of pointwise transition errors.

**Q12:** Bregman divergence: $D_g(x, y) = g(x) - g(y) - \langle \nabla g(y), x - y\rangle$. MD update: $x_{t+1} = \arg\min_{x \in \mathcal{C}} \eta\langle \nabla f(x_t), x\rangle + D_g(x, x_t)$. Full content: [[Bregman Divergence]] + [[Mirror Descent Analysis]].

**Q13:** Case split. If $z \geq 0$: $\alpha z \geq 0$ (since $\alpha \geq 0$), so $\mathrm{ReLU}(\alpha z) = \alpha z = \alpha \cdot \mathrm{ReLU}(z)$. If $z < 0$: $\alpha z \leq 0$, so $\mathrm{ReLU}(\alpha z) = 0 = \alpha \cdot 0 = \alpha \cdot \mathrm{ReLU}(z)$. (Note: fails for $\alpha < 0$.)

**Q14:** Original gradient: $\nabla_\phi \sum_i \ell_i$. New gradient: $\nabla_\phi \sum_i \ell_i + 2\lambda \phi$. The extra $2\lambda \phi$ "pulls weights toward zero" → "weight decay" — leads to update $\phi \leftarrow (1 - 2\eta\lambda)\phi - \eta \nabla \sum \ell_i$.

**Q15:** Full statement and proof in [[Policy Gradient Theorem]]. Statement: $\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}_\gamma, a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a)] / (1-\gamma)$. Proof: log-derivative trick → $\nabla\log P_\theta(\tau) = \sum_t \nabla\log\pi_\theta(a_t|s_t)$ (transitions cancel) → causality → tower property gives $Q^\pi$.

**Q16:** EXP3 setting: [[Adversarial Bandits]] (loss vector adversarial; only $\ell_t(a_t)$ observed). Algorithm: Hedge weights $w_t$, sample $a_t \sim w_t/\sum w_t$, importance-weight loss $\tilde\ell_t(a) = \ell_t(a_t)/x_t(a_t) \cdot \mathbb{1}[a_t = a]$, multiplicative update $w_{t+1}(a) = w_t(a) e^{-\eta \tilde\ell_t(a)}$. Regret: $O(\sqrt{KT \log K})$ with $\eta = \sqrt{\log K/(KT)}$. Full content: [[EXP3]].

</details>

## Topic tags

Bandit-related: Q9 ([[Q-learning]]), Q10 ([[Epsilon-Greedy]]), Q11 ([[VI Generative Setting]]), Q12 ([[Bregman Divergence]] + [[Mirror Descent Analysis]]), Q15 ([[Policy Gradient Theorem]]), Q16 ([[EXP3]], [[Adversarial Bandits]]).

DL-related: Q1 (VAE), Q2 (backprop), Q3 (ELBO), Q4 (UAT), Q5 (GAN), Q6 (regularization), Q7 (MLE), Q8 (Transformer), Q13 (ReLU), Q14 (L2 regularization).
