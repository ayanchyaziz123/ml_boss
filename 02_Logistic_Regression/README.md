# Logistic Regression — Interview Questions & Answers

---

## Small Dataset Example

| Age | Income ($k) | Loan Amount ($k) | Defaulted (1=Yes) |
|-----|-------------|------------------|-------------------|
| 25 | 35 | 10 | 0 |
| 45 | 85 | 25 | 0 |
| 35 | 50 | 40 | 1 |
| 52 | 45 | 35 | 1 |
| 23 | 28 | 15 | 1 |
| 40 | 90 | 20 | 0 |
| 60 | 100 | 50 | 0 |
| 30 | 40 | 45 | 1 |

### Dataset Questions

**Q: You train a Logistic Regression on this dataset. For a 33-year-old with $42k income and $38k loan, the model outputs 0.72. What does this mean?**
> **A:** The model estimates a 72% probability of defaulting. Since 0.72 > 0.5 (default threshold), this person would be classified as **Defaulted = 1**.

**Q: If you lower the classification threshold from 0.5 to 0.3, what happens to precision and recall?**
> **A:** Recall increases (more defaults caught), but precision decreases (more false positives — people incorrectly flagged as defaulters). Use a lower threshold when missing a true default is costly (e.g., banking risk).

**Q: The dataset has 5 non-defaults and 3 defaults. Is this imbalanced? What would you do?**
> **A:** Mildly imbalanced (62.5% vs 37.5%). For small datasets, use `class_weight='balanced'` in sklearn. For larger imbalances, use SMOTE or collect more minority class samples.

---

## Questions & Answers

---

### Q1. What is Logistic Regression?

**A:** Logistic Regression is a supervised classification algorithm that predicts the probability of a binary outcome (0 or 1). Despite the name, it is a **classification** algorithm, not regression.

It uses the sigmoid function to squash the linear output into a probability between 0 and 1:
```
P(y=1) = 1 / (1 + e^(−z))   where z = β₀ + β₁x₁ + ... + βₙxₙ
```

---

### Q2. What is the sigmoid function?

**A:** The sigmoid function maps any real number to a value between 0 and 1:
```
σ(z) = 1 / (1 + e^(−z))
```
- As z → +∞, σ(z) → 1
- As z → −∞, σ(z) → 0
- At z = 0, σ(z) = 0.5

This makes it ideal for outputting probabilities.

---

### Q3. Why is Logistic Regression used for classification?

**A:** Linear Regression can output values outside [0,1], making it unsuitable for probability estimation. The sigmoid function constrains the output to (0,1), allowing direct probability interpretation. The decision boundary cleanly separates classes.

---

### Q4. What is a decision boundary?

**A:** The decision boundary is the line (or surface) that separates the two classes. In Logistic Regression:
```
Predict class 1 if P(y=1) ≥ threshold (default 0.5)
Predict class 0 if P(y=1) < threshold
```
The boundary itself is where `P = 0.5`, i.e., where `z = 0`.

Logistic Regression produces a **linear** decision boundary.

---

### Q5. What is log loss (cross-entropy loss)?

**A:** Log loss penalizes confident wrong predictions heavily:
```
Log Loss = −(1/n) Σ [yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
```
- If y=1 and ŷ=0.99 → loss ≈ 0.01 (good prediction)
- If y=1 and ŷ=0.01 → loss ≈ 4.6 (bad prediction, heavy penalty)

Lower log loss = better model.

---

### Q6. What is the difference between Logistic and Linear Regression?

**A:**
| | Linear Regression | Logistic Regression |
|---|---|---|
| Output | Continuous value | Probability (0–1) |
| Task | Regression | Classification |
| Loss function | MSE | Log loss (cross-entropy) |
| Activation | None | Sigmoid |
| Decision boundary | N/A | Linear |

---

### Q7. What is ROC-AUC?

**A:**
- **ROC curve** plots True Positive Rate (Recall) vs False Positive Rate at various thresholds
- **AUC** = Area Under the ROC Curve (0 to 1)
  - AUC = 1.0 → perfect classifier
  - AUC = 0.5 → random classifier
  - AUC = 0.0 → perfectly wrong

AUC is threshold-independent and works well for balanced datasets.

---

### Q8. What is precision and recall?

**A:**
```
Precision = TP / (TP + FP)   → Of all predicted positives, how many are correct?
Recall    = TP / (TP + FN)   → Of all actual positives, how many did we catch?
```
- **High precision needed:** Spam detection (avoid false alarms)
- **High recall needed:** Cancer detection (avoid missing disease)

**F1 Score** = harmonic mean of precision and recall:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

---

### Q9. What is regularization in Logistic Regression?

**A:** Regularization prevents overfitting by penalizing large coefficients.

- **L2 (Ridge):** `Cost = Log Loss + λ Σβᵢ²` — shrinks coefficients, default in sklearn
- **L1 (Lasso):** `Cost = Log Loss + λ Σ|βᵢ|` — can zero out coefficients (feature selection)

In sklearn, use parameter `C = 1/λ` — smaller C = stronger regularization.

```python
LogisticRegression(C=0.1, penalty='l2')  # strong regularization
LogisticRegression(C=100, penalty='l2')  # weak regularization
```

---

### Q10. How do you evaluate Logistic Regression?

**A:** Multiple metrics depending on the use case:
1. **Accuracy** — good for balanced datasets
2. **Precision / Recall / F1** — for imbalanced datasets
3. **AUC-ROC** — threshold-independent evaluation
4. **Log Loss** — measures calibration quality
5. **Confusion Matrix** — breakdown of TP, TN, FP, FN

---

### Q11. What is the softmax function?

**A:** Softmax extends sigmoid to multi-class classification:
```
softmax(zᵢ) = e^(zᵢ) / Σ e^(zⱼ)
```
Outputs a probability distribution over all classes that sums to 1.

---

### Q12. How does Logistic Regression handle multi-class classification?

**A:**
- **One-vs-Rest (OvR):** Train K binary classifiers, each class vs all others. Pick the class with highest probability.
- **Multinomial (Softmax):** Train one model with softmax output for all K classes simultaneously.

```python
LogisticRegression(multi_class='ovr')       # One-vs-Rest
LogisticRegression(multi_class='multinomial', solver='lbfgs')  # Softmax
```

---

### Q13. What is a confusion matrix?

**A:** A table showing the performance of a classification model:

```
                Predicted 0    Predicted 1
Actual 0    |      TN      |      FP      |
Actual 1    |      FN      |      TP      |
```
- **TP:** Correctly predicted positive
- **TN:** Correctly predicted negative
- **FP:** Incorrectly predicted positive (Type I error)
- **FN:** Incorrectly predicted negative (Type II error)

---

### Q14. What is class imbalance and how do you handle it?

**A:** Class imbalance is when one class has significantly more samples than another (e.g., 95% non-fraud, 5% fraud).

**Handling techniques:**
1. `class_weight='balanced'` in sklearn
2. **SMOTE** — create synthetic minority samples
3. **Undersampling** — remove majority class samples
4. **Oversampling** — duplicate minority class samples
5. Use **AUC-PR** instead of accuracy as evaluation metric

---

### Q15. What is the threshold in Logistic Regression?

**A:** The threshold converts probability to class label. Default = 0.5.

- **Lower threshold (e.g., 0.3):** Higher recall, lower precision → catches more positives
- **Higher threshold (e.g., 0.7):** Higher precision, lower recall → fewer false positives

Choose threshold based on business cost of FP vs FN using the Precision-Recall curve.

---

### Q16. What is L1 vs L2 regularization in Logistic Regression?

**A:**
| | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Penalty | `Σ\|βᵢ\|` | `Σβᵢ²` |
| Feature selection | Yes (some β → 0) | No |
| Solution | No closed-form | Closed-form |
| When to use | Sparse features | Dense features |

---

### Q17. What is the log-odds interpretation of coefficients?

**A:** Each coefficient βᵢ represents the change in **log-odds** of the positive class for a one-unit increase in xᵢ:
```
log(P/(1-P)) = β₀ + β₁x₁ + ... + βₙxₙ
```
`e^(βᵢ)` = the **odds ratio** — how much the odds of the positive class multiply for a one-unit increase in xᵢ.

---

### Q18. What is the F1 score?

**A:** F1 is the harmonic mean of precision and recall:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
It penalizes large differences between precision and recall. Best when you need to balance both. Range: 0 (worst) to 1 (best).

---

### Q19. Why can't we use MSE as the loss function for Logistic Regression?

**A:** MSE combined with sigmoid creates a non-convex loss function with many local minima, making optimization with gradient descent unreliable. Cross-entropy loss is convex with sigmoid, guaranteeing convergence to the global minimum.

---

### Q20. What is the difference between Logistic Regression and SVM?

**A:**
| | Logistic Regression | SVM |
|---|---|---|
| Output | Probability | Class label |
| Loss | Cross-entropy | Hinge loss |
| Decision boundary | Passes through center of data | Maximizes margin |
| Outlier sensitivity | High | Low (only support vectors matter) |
| Probabilistic | Yes | No (by default) |

---

## Quick Code Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

X = np.array([[25,35,10],[45,85,25],[35,50,40],[52,45,35],
              [23,28,15],[40,90,20],[60,100,50],[30,40,45]])
y = np.array([0, 0, 1, 1, 1, 0, 0, 1])

model = LogisticRegression()
model.fit(X, y)

preds = model.predict(X)
proba = model.predict_proba(X)[:, 1]

print(classification_report(y, preds))
print("AUC-ROC:", roc_auc_score(y, proba))
```
