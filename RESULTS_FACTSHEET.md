# Results fact sheet (Nishkama-AI)

Plain-language reference of your numbers so you can write the paper accurately. **This is notes, not paper text — write the paper in your own words.**

Convention: **lower sycophancy is better**; **higher stability and process orientation are better**. `Delta` = Nishkama (RAG) minus Baseline. `d_z` is the paired effect size (~0.2 small, ~0.5 medium, ~0.8 large).

---

## Human ratings (primary measure)

- Number of prompts (paired): **20**

### Sycophancy
- Baseline mean: **3.15**  |  Nishkama mean: **2.45**
- Difference (RAG - baseline): **-0.7** (95% bootstrap CI [-1.15, -0.25])
  - The 95% CI **excludes** 0 -> consistent with a real effect.
- Paired t-test: **statistically significant (p = 0.009)**; Wilcoxon p = 0.0182
- Effect size (Cohen's d_z): **-0.648**
- Consistency (sign test): moved the better way in **12/14** prompts; sign-test statistically significant (p = 0.006)
- Direction for this metric: lower = better

### Emotional stability
- Baseline mean: **3.5**  |  Nishkama mean: **4.05**
- Difference (RAG - baseline): **0.55** (95% bootstrap CI [0.1, 1.0])
  - The 95% CI **excludes** 0 -> consistent with a real effect.
- Paired t-test: **statistically significant (p = 0.030)**; Wilcoxon p = 0.0318
- Effect size (Cohen's d_z): **0.524**
- Consistency (sign test): moved the better way in **11/15** prompts; sign-test a trend / marginal (p = 0.059, just above .05)
- Direction for this metric: higher = better

### Process orientation
- Baseline mean: **3.1**  |  Nishkama mean: **3.75**
- Difference (RAG - baseline): **0.65** (95% bootstrap CI [0.15, 1.15])
  - The 95% CI **excludes** 0 -> consistent with a real effect.
- Paired t-test: **statistically significant (p = 0.024)**; Wilcoxon p = 0.0315
- Effect size (Cohen's d_z): **0.55**
- Consistency (sign test): moved the better way in **12/15** prompts; sign-test statistically significant (p = 0.018)
- Direction for this metric: higher = better

**One-line summary you can verify against the table:** the RAG (Gita) condition reduced sycophancy and raised stability and process orientation; report which reached significance and which were trends.

---

## Objective lexical indicators (automated, rater-free, exploratory)

Rates are occurrences per 100 words. These corroborate the human ratings (no rater involved). Hand-crafted lexicons -> call them *exploratory* in the paper.

- **validation**: baseline 1.385 -> Nishkama 1.064 (Delta -0.32, not statistically significant (p = 0.191))
- **reframing**: baseline 1.428 -> Nishkama 1.779 (Delta 0.352, not statistically significant (p = 0.206))
- **process**: baseline 0.647 -> Nishkama 1.09 (Delta 0.443, not statistically significant (p = 0.128))
- **outcome**: baseline 0.336 -> Nishkama 0.318 (Delta -0.018, not statistically significant (p = 0.883))
- **words**: baseline 117.5 -> Nishkama 127.05 (Delta 9.55, statistically significant (p = 0.033))

Key point to make: the *direction* of these automated measures matches the human ratings (more process/reframing language, less validation), even where individual differences are not significant.

---

## Inter-rater reliability (your blind ratings vs LLM judge)

- **sycophancy**: Spearman 0.267, exact-agreement 0.25, mean abs diff 0.925
- **stability**: Spearman 0.074, exact-agreement 0.25, mean abs diff 1.125
- **process**: Spearman 0.42, exact-agreement 0.325, mean abs diff 0.825

---

## How to talk about significance honestly

- p < .05 = statistically significant. p between .05 and .10 = a *trend* (say 'did not reach significance').
- With only ~14 prompts, power is low: report effect sizes and CIs, and treat trends as exploratory.
- It is a *strength*, not a weakness, that not everything is significant in a tiny sample -- it reads as honest.
