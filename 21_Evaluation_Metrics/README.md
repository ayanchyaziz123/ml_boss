# Evaluation Metrics — Interview Questions & Answers

---

## Small Dataset Example

| Actual | Predicted | Probability (P=1) |
|--------|-----------|-------------------|
| 1 | 1 | 0.92 |
| 0 | 0 | 0.08 |
| 1 | 0 | 0.35 |
| 0 | 1 | 0.78 |
| 1 | 1 | 0.88 |
| 0 | 0 | 0.12 |
| 1 | 1 | 0.95 |
| 0 | 0 | 0.22 |

Threshold = 0.5 | TP=3, TN=4, FP=1, FN=1

### Dataset Questions

**Q: Calculate accuracy, precision, recall, and F1 for this dataset.**
> **A:**
> - Accuracy = (TP+TN)/(TP+TN+FP+FN) = (3+4)/8 = **0.875**
> - Precision = TP/(TP+FP) = 3/4 = **0.75**
> - Recall = TP/(TP+FN) = 3/4 = **0.75**
> - F1 = 2×(0.75×0.75)/(0.75+0.75) = **0.75**

**Q: If you lower the threshold from 0.5 to 0.3, row 3 (probability=0.35) is now predicted positive. What changes?**
> **A:** That sample (Actual=1, was FN) becomes TP. TP=4, FN=0. Recall = 4/4 = **1.0**, but also row with probability=0.35 becomes positive. Precision = 4/5 = **0.8**. Recall improved, precision slightly decreased.

**Q: Is accuracy a good metric here if the dataset had 90 negatives and 10 positives?**
> **A:** No. A model predicting all-negative achieves 90% accuracy but catches zero positives (useless). Use **F1 score**, **AUC-PR**, or **Recall** for imbalanced datasets. Accuracy is misleading when classes are unbalanced.

---

## Questions & Answers

---

### Q1. What is accuracy?

**A:** Accuracy is the fraction of total predictions that are correct:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use:** Balanced classes.
**When NOT to use:** Imbalanced classes (99% negative → 99% accuracy by predicting all negative).

---

### Q2. What is precision?

**A:** Precision measures how many of the predicted positives are actually positive:
```
Precision = TP / (TP + FP)
```

"Of all the times we said YES, how often were we right?"

**Optimize when:** False positives are costly. Example: spam detection (don't want real emails in spam folder), fraud alerts (don't want to block legitimate transactions).

---

### Q3. What is recall (sensitivity)?

**A:** Recall measures how many actual positives were correctly identified:
```
Recall = TP / (TP + FN)
```

"Of all the actual YES cases, how many did we catch?"

**Optimize when:** False negatives are costly. Example: cancer detection (missing a tumor is worse than a false alarm), fraud detection (missing fraud is costly).

---

### Q4. What is the F1 score?

**A:** F1 is the harmonic mean of precision and recall — balances both:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
     = 2TP / (2TP + FP + FN)
```

Range: 0–1 (1 is perfect). Penalizes extreme imbalance between precision and recall. Use when both false positives and false negatives are important.

---

### Q5. What is the F-beta score?

**A:** F-beta generalizes F1 to weight recall β times more than precision:
```
Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```
- β=1: F1 (equal weight)
- β=2: Recall twice as important (medical diagnosis)
- β=0.5: Precision twice as important (information retrieval)

```python
from sklearn.metrics import fbeta_score
fbeta_score(y_true, y_pred, beta=2)
```

---

### Q6. What is AUC-ROC?

**A:** AUC-ROC measures discriminative ability across all thresholds:
- **ROC curve:** Plots TPR (recall) vs FPR at each threshold
- **AUC:** Area Under the ROC curve

```
TPR = TP / (TP + FN) = Recall
FPR = FP / (FP + TN)
```

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC = 0.7–0.8: Acceptable
- AUC ≥ 0.9: Excellent

```python
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_true, y_prob)
```

---

### Q7. What is AUC-PR (Precision-Recall curve)?

**A:** AUC-PR plots Precision vs Recall at each threshold:

- Better than AUC-ROC for **imbalanced datasets**
- AUC-ROC can look good even when model is poor for minority class
- AUC-PR focuses entirely on the minority (positive) class

```python
from sklearn.metrics import average_precision_score, precision_recall_curve
ap = average_precision_score(y_true, y_prob)
```

Rule: If classes are balanced, use AUC-ROC. If heavily imbalanced, use AUC-PR.

---

### Q8. What is a confusion matrix?

**A:**
```
                    Predicted Negative    Predicted Positive
Actual Negative  |       TN           |        FP          |
Actual Positive  |       FN           |        TP          |
```

All four cells matter:
- **TP:** Correctly predicted positive
- **TN:** Correctly predicted negative
- **FP:** Predicted positive, actually negative (Type I error)
- **FN:** Predicted negative, actually positive (Type II error)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
```

---

### Q9. What is True Positive Rate (TPR) and False Positive Rate (FPR)?

**A:**
```
TPR (Sensitivity, Recall) = TP / (TP + FN)  → % of actual positives caught
FPR (Fall-out)           = FP / (FP + TN)  → % of actual negatives incorrectly flagged
Specificity              = TN / (TN + FP) = 1 − FPR
```

ROC curve: trade-off between TPR and FPR as threshold changes.

---

### Q10. What is Mean Absolute Error (MAE)?

**A:**
```
MAE = (1/n) Σ|yᵢ − ŷᵢ|
```

- Average of absolute errors
- **Robust to outliers** — doesn't square errors
- Same unit as the target variable
- Minimized by median prediction

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

---

### Q11. What is Mean Squared Error (MSE)?

**A:**
```
MSE = (1/n) Σ(yᵢ − ŷᵢ)²
```

- Squares errors → large errors penalized heavily
- Sensitive to outliers
- Differentiable → useful as loss function
- Units: squared units of target

**Minimized by:** Mean prediction

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

---

### Q12. What is RMSE?

**A:** Root Mean Squared Error = √MSE:
```
RMSE = √[(1/n) Σ(yᵢ − ŷᵢ)²]
```

- Same unit as target (interpretable)
- Penalizes large errors more than MAE
- Most common regression metric

**Comparison:** If RMSE >> MAE, the model has large outlier errors. If RMSE ≈ MAE, errors are uniformly distributed.

---

### Q13. What is R-squared (R²)?

**A:** R² measures the proportion of variance in the target explained by the model:
```
R² = 1 − (SS_res / SS_tot)
SS_res = Σ(yᵢ − ŷᵢ)²  (residual sum of squares)
SS_tot = Σ(yᵢ − ȳ)²   (total sum of squares)
```

- R²=1: Perfect fit
- R²=0: Model predicts mean of y (no better than baseline)
- R²<0: Model is worse than predicting the mean (bad model)

---

### Q14. What is log loss?

**A:** Log loss (cross-entropy loss) measures quality of probabilistic predictions:
```
Log Loss = −(1/n) Σ[yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
```

- Lower is better (0 = perfect)
- Heavily penalizes confident wrong predictions
- Measures **calibration** — are predicted probabilities accurate?

```python
from sklearn.metrics import log_loss
ll = log_loss(y_true, y_prob)
```

---

### Q15. What is MAPE?

**A:** Mean Absolute Percentage Error:
```
MAPE = (100/n) Σ|yᵢ − ŷᵢ| / |yᵢ|
```

- Interpretable: "5% MAPE = predictions off by 5% on average"
- **Problem:** Division by zero if yᵢ=0; biased for small values
- Better alternative: **SMAPE** (Symmetric MAPE) or **MASE**

```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

---

### Q16. When do you use accuracy vs F1 score?

**A:**
- **Accuracy:** When classes are balanced AND each error type has equal cost
- **F1:** When classes are imbalanced OR false positives and false negatives have different costs
- **Precision:** When false positives are expensive (spam filter, recommendation)
- **Recall:** When false negatives are expensive (cancer detection, fraud)

General rule: Use F1 or AUC for imbalanced data (fraud, disease detection, anomaly detection).

---

### Q17. What is the difference between micro and macro averaging?

**A:**
- **Macro avg:** Calculate metric for each class independently, then average
  - Equal weight per class — useful when small classes matter equally
  - `average='macro'` in sklearn

- **Micro avg:** Pool all TP, FP, FN across all classes, then calculate
  - Dominated by majority classes
  - `average='micro'` in sklearn

- **Weighted avg:** Weighted by class support (sample count)
  - `average='weighted'` in sklearn

```python
precision_score(y_true, y_pred, average='macro')
```

---

### Q18. What is top-k accuracy?

**A:** For multi-class classification, top-k accuracy counts a prediction as correct if the true class is in the top k predictions:

```python
from sklearn.metrics import top_k_accuracy_score
# Correct if true label is in top 3 predicted probabilities
top3_acc = top_k_accuracy_score(y_true, y_prob, k=3)
```

Used in image classification (ImageNet: top-1 and top-5 accuracy), recommendation systems (top-10 hit rate).

---

### Q19. What is the difference between Type I and Type II error?

**A:**
- **Type I error (α, False Positive):** Reject a true null hypothesis — "False alarm"
  - Example: Diagnosing healthy person as sick
  - Controlled by significance level α (p-value threshold)

- **Type II error (β, False Negative):** Fail to reject a false null hypothesis — "Miss"
  - Example: Missing a sick person's diagnosis
  - Related to statistical power = 1 − β

In ML terms: FP = Type I, FN = Type II.

---

### Q20. What metrics are used for evaluating generative models?

**A:**

| Task | Metric |
|---|---|
| Image generation | FID (Fréchet Inception Distance), IS (Inception Score) |
| Translation | BLEU |
| Summarization | ROUGE |
| Text generation quality | Perplexity |
| LLM alignment | RLHF reward scores, GPT-4 judge |
| Similarity | BERTScore, BLEURT |
| Information retrieval | NDCG, MRR, MAP |

---

## Quick Code Example

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, log_loss,
                              confusion_matrix, classification_report)
import numpy as np

y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 1, 0, 1, 0])
y_prob = np.array([0.92, 0.08, 0.35, 0.78, 0.88, 0.12, 0.95, 0.22])

print("=== Classification Metrics ===")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_true, y_prob):.4f}")
print(f"Log Loss:  {log_loss(y_true, y_prob):.4f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=['Negative','Positive']))

# Regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_reg_true = np.array([3.0, -0.5, 2.0, 7.0])
y_reg_pred = np.array([2.5,  0.0, 2.0, 8.0])

print("\n=== Regression Metrics ===")
print(f"MAE:  {mean_absolute_error(y_reg_true, y_reg_pred):.4f}")
print(f"MSE:  {mean_squared_error(y_reg_true, y_reg_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_reg_true, y_reg_pred)):.4f}")
print(f"R²:   {r2_score(y_reg_true, y_reg_pred):.4f}")
```
