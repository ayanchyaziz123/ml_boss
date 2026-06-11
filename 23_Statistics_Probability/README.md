# Statistics & Probability — Interview Questions & Answers

---

## Small Dataset Example

| Student | Test Score | Hours Studied | Passed? |
|---------|-----------|---------------|---------|
| A | 72 | 6 | Yes |
| B | 45 | 2 | No |
| C | 88 | 9 | Yes |
| D | 55 | 4 | No |
| E | 92 | 10 | Yes |
| F | 61 | 5 | Yes |
| G | 38 | 1 | No |
| H | 78 | 7 | Yes |

### Dataset Questions

**Q: Calculate mean, median, and standard deviation of Test Scores.**
> **A:**
> - Mean = (72+45+88+55+92+61+38+78) / 8 = 529/8 = **66.1**
> - Sorted: 38, 45, 55, 61, 72, 78, 88, 92 → Median = (61+72)/2 = **66.5**
> - Variance = Σ(xᵢ−μ)² / (n-1) ≈ 378.4 → SD = √378.4 ≈ **19.5**

**Q: Is there a correlation between Hours Studied and Test Score? Calculate Pearson correlation.**
> **A:** Hours [6,2,9,4,10,5,1,7], Scores [72,45,88,55,92,61,38,78]. These move together consistently → strong positive correlation. Pearson r ≈ **0.99** (nearly perfect linear relationship in this toy data).

**Q: A student scores 92. What is their z-score? What does it mean?**
> **A:** z = (92 − 66.1) / 19.5 ≈ **1.33**. This student scored 1.33 standard deviations above the mean — approximately in the top 9% of the class (using normal distribution table).

---

## Questions & Answers

---

### Q1. What is probability?

**A:** Probability is a number between 0 and 1 that quantifies how likely an event is to occur:
- P(A) = 0: Event A is impossible
- P(A) = 1: Event A is certain
- P(A) = 0.5: Event A occurs half the time

**Axioms:**
1. P(A) ≥ 0
2. P(sample space) = 1
3. P(A ∪ B) = P(A) + P(B) for mutually exclusive A, B

---

### Q2. What is conditional probability?

**A:** Conditional probability is the probability of event A given that event B has occurred:
```
P(A|B) = P(A ∩ B) / P(B)
```

Example: P(Pass | Hours > 5) = P(Pass AND Hours > 5) / P(Hours > 5)

In our dataset: Hours > 5 cases: {A,C,E,H} = 4 students, all passed.
P(Pass | Hours > 5) = 4/4 = **1.0**

---

### Q3. What is Bayes' Theorem?

**A:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

Bayesian interpretation: Update prior belief P(A) after observing evidence B to get posterior P(A|B).

ML application: Naive Bayes classifier, Bayesian optimization, probabilistic graphical models.

Example: P(Disease | Test+) = P(Test+ | Disease) × P(Disease) / P(Test+)

---

### Q4. What is mean, median, and mode?

**A:**
- **Mean (μ):** Arithmetic average = Σxᵢ/n — sensitive to outliers
- **Median:** Middle value when sorted — robust to outliers
- **Mode:** Most frequent value — used for categorical data

```python
import numpy as np
from scipy import stats

scores = [72, 45, 88, 55, 92, 61, 38, 78]
print("Mean:", np.mean(scores))       # 66.1
print("Median:", np.median(scores))   # 66.5
print("Mode:", stats.mode(scores))
```

**Rule:** If mean > median, distribution is right-skewed. If mean < median, left-skewed.

---

### Q5. What is variance and standard deviation?

**A:**
```
Variance (σ²) = Σ(xᵢ − μ)² / (n−1)   # sample variance
Std Dev (σ)   = √(Variance)
```

- Variance: average squared deviation from mean — in squared units
- Std Dev: average deviation from mean — in original units (interpretable)

Standard deviation is used more in practice because it has the same units as the data.

---

### Q6. What is a normal distribution?

**A:** The normal (Gaussian) distribution is the classic bell curve:
```
f(x) = (1/σ√(2π)) × exp(−(x−μ)²/(2σ²))
```

Properties:
- Symmetric about the mean
- 68% data within 1σ, 95% within 2σ, 99.7% within 3σ
- Fully described by mean μ and standard deviation σ
- Central Limit Theorem: Sample means are normally distributed for large n

---

### Q7. What is a probability distribution?

**A:** A probability distribution describes the probability of each possible outcome.

**Discrete:** Probability Mass Function (PMF)
- Bernoulli: Success/failure (coin flip) — P(X=1) = p
- Binomial: Count of successes in n trials
- Poisson: Count of events in a time interval

**Continuous:** Probability Density Function (PDF)
- Normal: Bell curve
- Exponential: Time until event
- Uniform: Equal probability over range

---

### Q8. What is a hypothesis test?

**A:** A hypothesis test evaluates whether observed data provides sufficient evidence against a null hypothesis H₀:

1. State H₀ (null) and H₁ (alternative)
2. Choose significance level α (e.g., 0.05)
3. Compute test statistic
4. Calculate p-value
5. If p-value < α: reject H₀ (statistically significant)

Example: H₀: "New model accuracy = old model accuracy", H₁: "New model > old"

---

### Q9. What is the p-value?

**A:** The p-value is the probability of observing results at least as extreme as the actual results, **assuming H₀ is true**:

- Small p-value (< 0.05): Strong evidence against H₀ → reject H₀
- Large p-value (≥ 0.05): Insufficient evidence → fail to reject H₀

**Common misconception:** p-value is NOT the probability that H₀ is true.

---

### Q10. What is a confidence interval?

**A:** A 95% confidence interval gives a range that, if we repeated the experiment many times, would contain the true parameter 95% of the time:

```
CI = x̄ ± z × (σ/√n)

For 95%: z = 1.96
For 99%: z = 2.576
```

Example: Sample mean = 66.1, σ = 19.5, n = 8
95% CI = 66.1 ± 1.96 × (19.5/√8) = [52.6, 79.6]

---

### Q11. What is the Central Limit Theorem (CLT)?

**A:** The CLT states that the distribution of sample means approaches a normal distribution as sample size increases, regardless of the population distribution:

```
X̄ ~ N(μ, σ²/n) as n → ∞
```

Rule of thumb: n ≥ 30 is often sufficient.

**Why it matters:** Most statistical tests assume normality — CLT justifies applying them to non-normal data with large enough samples.

---

### Q12. What is the Law of Large Numbers?

**A:** As the number of trials increases, the sample mean converges to the true population mean:

```
X̄_n → μ as n → ∞
```

Example: Flip a fair coin 10 times → might get 60% heads. Flip 10,000 times → very close to 50%.

Guarantees that estimates from data converge to true values with enough data.

---

### Q13. What is correlation?

**A:** Pearson correlation measures the linear relationship between two variables:
```
r = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / √[Σ(xᵢ−x̄)² × Σ(yᵢ−ȳ)²]
```

- r = +1: Perfect positive linear relationship
- r = -1: Perfect negative linear relationship
- r = 0: No linear relationship

```python
import numpy as np
r = np.corrcoef(hours_studied, test_scores)[0, 1]
```

---

### Q14. What is covariance?

**A:** Covariance measures how two variables change together:
```
Cov(X,Y) = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / (n−1)
```

- Positive: Variables increase together
- Negative: One increases as the other decreases
- 0: No linear relationship

**Limitation:** Scale-dependent — hard to interpret magnitude. **Correlation = normalized covariance**.

```
r = Cov(X,Y) / (σ_X × σ_Y)
```

---

### Q15. What is the difference between correlation and causation?

**A:** Correlation measures statistical association; causation implies one variable directly causes another.

**Correlation ≠ Causation examples:**
- Ice cream sales and drowning rates are correlated (both increase in summer) — confounded by season
- Shoe size and reading ability in children are correlated — both caused by age

In ML: Correlated features are useful for prediction even without causal relationships. But for interventions (changing X to cause change in Y), you need causal understanding.

---

### Q16. What is a t-test?

**A:** A t-test compares means to determine if differences are statistically significant:

- **One-sample:** Is sample mean different from a known value?
- **Two-sample (independent):** Are means of two groups different?
- **Paired:** Are means of matched pairs different?

```python
from scipy import stats

# Two-sample t-test
control = [72, 45, 55, 38]
treatment = [88, 92, 61, 78]
t_stat, p_value = stats.ttest_ind(control, treatment)
print(f"t={t_stat:.3f}, p={p_value:.3f}")
```

Used when: comparing A/B test results, model performance between two methods.

---

### Q17. What is ANOVA?

**A:** ANOVA (Analysis of Variance) tests whether means of 3+ groups are significantly different:

H₀: μ₁ = μ₂ = μ₃ = ... (all group means are equal)

```python
from scipy import stats
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

Used when comparing multiple algorithms, multiple feature configurations, or multiple model variants.

---

### Q18. What is chi-square test?

**A:** Chi-square (χ²) tests independence between categorical variables:

```
χ² = Σ (Observed − Expected)² / Expected
```

Used for:
- Feature selection (dependency between feature and target)
- Fairness testing (model predictions vs protected attributes)
- A/B testing with categorical outcomes

```python
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

---

### Q19. What is Type I and Type II error?

**A:**
| | Reality: H₀ True | Reality: H₀ False |
|---|---|---|
| Decision: Reject H₀ | Type I Error (α) False Positive | Correct (Power = 1−β) |
| Decision: Fail to Reject H₀ | Correct (1−α) | Type II Error (β) False Negative |

- **Type I (α=0.05):** Concluding there's an effect when there isn't (false alarm)
- **Type II (β):** Failing to detect a real effect (miss)
- **Power (1−β):** Probability of detecting a real effect

---

### Q20. What is statistical significance vs practical significance?

**A:**
- **Statistical significance:** p < 0.05 — the effect is unlikely due to chance
- **Practical significance:** The effect is large enough to matter in practice

With millions of samples, even tiny differences become statistically significant but may be practically meaningless.

Always report **effect size** (Cohen's d, R²) alongside p-values:

```python
# Cohen's d for t-test
d = (mean1 - mean2) / pooled_std
# |d| < 0.2: Small, 0.2-0.5: Medium, > 0.8: Large
```

---

## Quick Code Example

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

scores = np.array([72, 45, 88, 55, 92, 61, 38, 78])
hours  = np.array([6,   2,  9,  4, 10,  5,  1,  7])

print("=== Descriptive Statistics ===")
print(f"Mean:    {np.mean(scores):.2f}")
print(f"Median:  {np.median(scores):.2f}")
print(f"Std Dev: {np.std(scores, ddof=1):.2f}")
print(f"Variance:{np.var(scores, ddof=1):.2f}")
print(f"Min: {scores.min()}, Max: {scores.max()}")

print("\n=== Correlation ===")
r, p = stats.pearsonr(hours, scores)
print(f"Pearson r = {r:.4f}, p-value = {p:.4f}")

print("\n=== Z-scores ===")
z_scores = (scores - np.mean(scores)) / np.std(scores, ddof=1)
for s, z in zip(scores, z_scores):
    print(f"Score {s}: z = {z:.2f}")

print("\n=== Confidence Interval for Mean ===")
n = len(scores)
se = np.std(scores, ddof=1) / np.sqrt(n)
ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=np.mean(scores), scale=se)
print(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

print("\n=== Hypothesis Test ===")
# t-test: Is mean significantly different from 65?
t_stat, p_val = stats.ttest_1samp(scores, 65)
print(f"t-stat = {t_stat:.3f}, p = {p_val:.3f}")
print("Reject H₀?" , "Yes" if p_val < 0.05 else "No")

print("\n=== Normality Test ===")
stat, p_norm = stats.shapiro(scores)
print(f"Shapiro-Wilk: stat={stat:.4f}, p={p_norm:.4f}")
print("Normal distribution?", "Yes" if p_norm > 0.05 else "No")
```
