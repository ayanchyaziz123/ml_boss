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

---

## Additional Questions & Answers (Q21–Q50)

---

### Q21. What is the gradient of the log loss with respect to weights?

**A:** For a single sample, the gradient of binary cross-entropy with respect to weights is:
```
∂L/∂w = (ŷ − y) × x
```
Where `ŷ = σ(wᵀx)` is the predicted probability and y is the true label.

This is elegantly simple — the gradient is just the prediction error multiplied by the input. This is why gradient descent works cleanly for Logistic Regression.

---

### Q22. What is the difference between Logistic Regression and a single-layer neural network?

**A:** They are mathematically identical for binary classification:
- Both compute: `z = wᵀx + b`
- Both apply sigmoid: `ŷ = σ(z)`
- Both minimize cross-entropy loss

The only differences are terminology and ecosystem. Logistic Regression = neural network with zero hidden layers and one output neuron with sigmoid activation.

---

### Q23. What are the solvers available in sklearn's LogisticRegression and when do you use each?

**A:**
| Solver | Best For |
|---|---|
| `lbfgs` | Default — small/medium datasets, L2 only |
| `liblinear` | Small datasets, L1 or L2, binary |
| `saga` | Large datasets, all penalties (L1, L2, Elastic Net), sparse data |
| `newton-cg` | L2 only, medium datasets |
| `sag` | Large datasets, L2 only |

```python
LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5)
```

---

### Q24. How does feature scaling affect Logistic Regression?

**A:** Logistic Regression is sensitive to feature scale when using gradient-based optimization:
- Features with large scales dominate gradient updates
- Model may converge slowly or to suboptimal solution
- Regularization (L1/L2) penalizes larger coefficients unfairly if features aren't scaled

**Fix:** Always standardize features (`StandardScaler`) before training Logistic Regression, especially with regularization.

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
```

---

### Q25. What is the relationship between Logistic Regression and Naive Bayes?

**A:** They are related — Naive Bayes (Gaussian) and Logistic Regression model the same family of decision boundaries, but:

- **Naive Bayes (generative):** Models P(X|y) and P(y) separately, then uses Bayes' theorem
- **Logistic Regression (discriminative):** Directly models P(y|X)

When the Naive Bayes independence assumption holds perfectly, both give the same decision boundary. Logistic Regression requires more data but makes fewer assumptions.

---

### Q26. What is the maximum likelihood estimation (MLE) interpretation of Logistic Regression?

**A:** Training Logistic Regression via cross-entropy loss is equivalent to **Maximum Likelihood Estimation** — we find weights that maximize the likelihood of observing the training labels given the inputs:

```
L(w) = Π P(yᵢ|xᵢ; w) = Π ŷᵢ^yᵢ × (1−ŷᵢ)^(1−yᵢ)
```

Taking the log:
```
log L(w) = Σ [yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
```

Maximizing log-likelihood = minimizing cross-entropy loss. The two are exactly equivalent.

---

### Q27. What is perfect separation and why does it cause problems?

**A:** Perfect separation (complete separation) occurs when a feature or combination of features perfectly predicts the class — all class-1 samples have x > c and all class-0 samples have x < c.

**Problem:** The MLE estimate for that feature's coefficient → ±∞. The optimization never converges. Sklearn may silently return very large coefficients.

**Detection:** Look for very large coefficients (|β| > 10) or convergence warnings.

**Fix:** Use Firth's penalized logistic regression, Ridge regularization (L2), or remove the perfectly separating feature.

---

### Q28. What is Platt scaling?

**A:** Platt scaling is a post-processing method that calibrates the probability outputs of a classifier (like SVM) by fitting a Logistic Regression on the raw scores:

```
P(y=1 | score) = σ(A × score + B)
```

The parameters A and B are learned on a held-out validation set. This makes poorly calibrated scores (like SVM's decision function) into proper probabilities.

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(svm_model, method='sigmoid', cv=5)
```

---

### Q29. What is a calibration curve (reliability diagram)?

**A:** A calibration curve plots predicted probabilities against actual observed frequencies to check if probabilities are trustworthy:

- **Well-calibrated:** Points lie on the diagonal (ŷ=0.7 → 70% of those samples are actually positive)
- **Overconfident:** Curve bows toward corners
- **Underconfident:** Curve bows toward center

```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true, 's-')
plt.plot([0,1],[0,1], '--', label='Perfect calibration')
```

---

### Q30. What is the difference between predict() and predict_proba() in sklearn?

**A:**
- **`predict(X)`:** Returns hard class labels (0 or 1) using the default threshold (0.5)
- **`predict_proba(X)`:** Returns probability for each class → shape (n_samples, n_classes)

```python
model.predict(X)         # → [0, 1, 1, 0, ...]
model.predict_proba(X)   # → [[0.82, 0.18], [0.23, 0.77], ...]
# proba[:, 1] = P(class=1) for each sample
```

Use `predict_proba` when you need:
- Custom thresholds
- Ranking predictions
- Probability calibration
- Input to a downstream model

---

### Q31. How do you handle multicollinear features in Logistic Regression?

**A:** Multicollinearity makes coefficient estimates unstable and hard to interpret (similar to Linear Regression):

**Detection:**
- Correlation matrix between features
- VIF > 10 for any feature

**Fix:**
1. Remove one of the correlated features
2. Use Ridge (L2) regularization — shrinks correlated coefficients toward each other
3. Apply PCA to reduce to uncorrelated components before training

---

### Q32. What is the Hosmer-Lemeshow test?

**A:** The Hosmer-Lemeshow test checks the goodness-of-fit of a Logistic Regression model:
1. Divide predictions into 10 equal-frequency bins
2. Compare observed vs expected event rates in each bin
3. Compute χ² statistic

- p > 0.05: Model fits well
- p < 0.05: Model is a poor fit to the data

This is the logistic regression equivalent of checking residuals in Linear Regression.

---

### Q33. What is McFadden's Pseudo R²?

**A:** There is no direct R² for Logistic Regression (no SS decomposition). McFadden's pseudo-R² is a common substitute:

```
R²_McFadden = 1 − (log L_model / log L_null)
```

- `log L_model`: Log-likelihood of fitted model
- `log L_null`: Log-likelihood of intercept-only model

Range: 0 to 1. Values of 0.2–0.4 indicate excellent fit (much lower than R² in linear regression).

---

### Q34. What is the difference between binary and ordinal logistic regression?

**A:**
| | Binary Logistic Regression | Ordinal Logistic Regression |
|---|---|---|
| Target | 2 classes (0/1) | 3+ ordered classes (Low/Med/High) |
| Model | One sigmoid | Cumulative logits |
| Assumption | — | Proportional odds |
| Example | Pass/Fail | Ratings (1–5 stars) |

Ordinal logistic uses the **proportional odds model** — assumes the same feature coefficients apply at each ordinal boundary.

---

### Q35. What is the effect of adding irrelevant features to Logistic Regression?

**A:**
- Without regularization: Model overfits. Irrelevant features get non-zero coefficients by fitting noise
- With L2 (Ridge): Irrelevant features' coefficients shrink toward 0 — model is more robust
- With L1 (Lasso): Irrelevant features' coefficients are driven exactly to 0 — automatic feature selection

This is why regularization is especially important when number of features is large relative to samples.

---

### Q36. How do you interpret coefficients when features are on different scales?

**A:** Raw coefficients reflect the effect of a **one-unit change** in the feature. If features have different scales, raw coefficients are not directly comparable.

**Solution:** Standardize features before training. Then:
- All coefficients are in terms of "one standard deviation change"
- Coefficients can be compared directly (larger |β| = stronger effect)
- Called **standardized coefficients**

```python
pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
pipe.fit(X, y)
# Coefficients are now standardized
```

---

### Q37. What is stratified K-Fold cross-validation and why is it important for Logistic Regression?

**A:** Stratified K-Fold ensures each fold has the same class proportion as the full dataset:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
```

**Why critical for Logistic Regression:** If one fold accidentally gets all negatives and no positives, the model can't learn the minority class. Stratification guarantees balanced splits, especially important for imbalanced datasets.

---

### Q38. What is the difference between macro-averaged and weighted-averaged F1?

**A:**
```python
from sklearn.metrics import f1_score

# Macro: equal weight per class
f1_macro = f1_score(y_true, y_pred, average='macro')

# Weighted: weighted by class support (number of samples)
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# Micro: global TP, FP, FN pooled across classes
f1_micro = f1_score(y_true, y_pred, average='micro')
```

For **imbalanced datasets**: use macro F1 when all classes matter equally, weighted F1 when larger classes matter more.

---

### Q39. What is cost-sensitive learning in Logistic Regression?

**A:** Cost-sensitive learning adjusts the model for unequal misclassification costs:

```python
# Method 1: class_weight
model = LogisticRegression(class_weight={0: 1, 1: 10})  # FN 10x more costly than FP

# Method 2: auto-balance
model = LogisticRegression(class_weight='balanced')

# Method 3: sample_weight
model.fit(X, y, sample_weight=sample_weights)
```

Useful when: missing a fraud transaction is 100× more costly than a false alarm.

---

### Q40. How does Logistic Regression handle missing values?

**A:** Logistic Regression (like most sklearn models) does not handle missing values natively — will throw a `ValueError`.

**Pre-processing approaches:**
```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression())
])
```

Also add a missingness indicator column: the fact that a value is missing can itself be predictive.

---

### Q41. What is the learning curve and how do you use it?

**A:** A learning curve plots training and validation error as a function of training set size — diagnoses bias vs variance:

```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
```

- **High bias (underfitting):** Both train and val error high, small gap → more features, complex model
- **High variance (overfitting):** Train error low, val error high, large gap → more data, regularization

---

### Q42. What is the difference between discriminative and generative classifiers?

**A:**
| | Discriminative | Generative |
|---|---|---|
| Models | P(y\|X) directly | P(X\|y) and P(y), derives P(y\|X) |
| Examples | Logistic Regression, SVM | Naive Bayes, LDA, GMM |
| Data efficiency | Needs more data | Works with less data |
| Out-of-distribution | Doesn't know | Can detect (low P(X)) |
| New sample generation | Cannot | Can generate samples |

Logistic Regression is discriminative — it directly learns the boundary between classes.

---

### Q43. What is one-vs-one (OvO) vs one-vs-rest (OvR) for multi-class?

**A:**
| | One-vs-Rest (OvR) | One-vs-One (OvO) |
|---|---|---|
| Classifiers trained | K | K(K-1)/2 |
| Each classifier | Class k vs all others | Class k vs class j |
| Prediction | Max score across classifiers | Majority vote |
| Speed | Faster | Slower (more classifiers) |
| Accuracy | Slightly lower | Slightly higher |

For K=10 classes: OvR trains 10 classifiers, OvO trains 45. Default in sklearn `LogisticRegression` is OvR.

---

### Q44. How would you tune Logistic Regression hyperparameters?

**A:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])

param_grid = {
    'lr__C':       [0.001, 0.01, 0.1, 1, 10, 100],
    'lr__penalty': ['l1', 'l2'],
    'lr__solver':  ['liblinear', 'saga']
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best AUC:", gs.best_score_)
```

Key hyperparameters: `C` (regularization strength) and `penalty` (L1 vs L2).

---

### Q45. What is the difference between AUC-ROC and log loss as evaluation metrics?

**A:**
| | AUC-ROC | Log Loss |
|---|---|---|
| Measures | Ranking quality (discrimination) | Probability calibration |
| Threshold-dependent | No | No |
| Sensitive to class imbalance | Less | More |
| Range | 0–1 (higher=better) | 0–∞ (lower=better) |
| Penalizes confident wrong predictions | No | Yes (heavily) |
| Use when | Need to compare model discrimination | Need well-calibrated probabilities |

High AUC + high log loss: model ranks well but is poorly calibrated (overconfident).

---

### Q46. What is the intercept term in Logistic Regression?

**A:** The intercept β₀ shifts the decision boundary. It captures the **baseline log-odds** when all features are 0:

```
log(P/(1-P)) = β₀   when all xᵢ = 0
→ P = σ(β₀) = prior probability of positive class
```

If class imbalance exists (5% positives), the intercept will be negative: `β₀ = log(0.05/0.95) ≈ −2.94`, reflecting the low base rate.

Setting `fit_intercept=False` forces the boundary through the origin — almost never appropriate.

---

### Q47. What is online learning in the context of Logistic Regression?

**A:** Online learning updates the model incrementally as new data arrives — no full retraining:

```python
from sklearn.linear_model import SGDClassifier

# SGDClassifier with log loss = Logistic Regression trained with SGD
model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01)

# Incremental fit
for batch_X, batch_y in data_stream:
    model.partial_fit(batch_X, batch_y, classes=[0, 1])
```

Use when:
- Data arrives continuously (streaming)
- Dataset too large to fit in memory
- Model must adapt to distribution drift over time

---

### Q48. What are some real-world applications of Logistic Regression?

**A:**
| Application | Target (y) | Features |
|---|---|---|
| Credit scoring | Default (0/1) | Income, debt, age |
| Medical diagnosis | Disease (0/1) | Lab values, symptoms |
| Email spam | Spam (0/1) | Word frequencies (TF-IDF) |
| Ad click prediction | Click (0/1) | User behavior, ad features |
| Churn prediction | Churned (0/1) | Usage, tenure, support calls |
| Fraud detection | Fraud (0/1) | Transaction amount, location |

Despite the availability of complex models, Logistic Regression is often the production baseline — fast, interpretable, and robust.

---

### Q49. What is the effect of correlated features on Logistic Regression coefficients?

**A:** Correlated features split the predictive effect between them, making individual coefficients unstable:

Example: If `income` and `credit_score` are 0.9 correlated:
- True effect: each unit increase in income → +0.5 log-odds
- With both in model: income → +0.8, credit_score → −0.3 (unstable, wrong signs)

**Fix:**
1. Remove one correlated feature
2. Use Ridge (L2) which distributes weight evenly between correlated features
3. Create a single combined feature

---

### Q50. How do you report Logistic Regression results in a professional setting?

**A:** A complete reporting template:

```
Model: Logistic Regression (L2, C=0.1, solver=lbfgs)
Dataset: 10,000 samples, 15 features, 8% positive class

Preprocessing: StandardScaler, mean imputation for missing values

Evaluation (5-fold stratified CV):
  - AUC-ROC:    0.847 ± 0.012
  - AUC-PR:     0.623 ± 0.018
  - F1 (macro): 0.791 ± 0.014
  - Log Loss:   0.312 ± 0.008

Top predictors (by |standardized coefficient|):
  1. credit_score:    −0.82 (higher score → lower default risk)
  2. debt_to_income:  +0.71 (higher ratio → higher default risk)
  3. loan_amount:     +0.45

Calibration: Good (Brier score = 0.07, calibration plot near diagonal)
```

---

## Extended Code Examples

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, log_loss,
                              calibration_curve, confusion_matrix)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

X = np.array([[25,35,10],[45,85,25],[35,50,40],[52,45,35],
              [23,28,15],[40,90,20],[60,100,50],[30,40,45]])
y = np.array([0, 0, 1, 1, 1, 0, 0, 1])

# Pipeline with scaling
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr',     LogisticRegression(C=1.0, penalty='l2', solver='lbfgs'))
])

# Stratified CV
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
auc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
f1_scores  = cross_val_score(pipe, X, y, cv=skf, scoring='f1')
print(f"AUC-ROC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
print(f"F1:      {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# Full fit for predictions
pipe.fit(X, y)
proba = pipe.predict_proba(X)[:, 1]
preds = pipe.predict(X)

# Comprehensive evaluation
print("\n=== Evaluation ===")
print(f"AUC-ROC:   {roc_auc_score(y, proba):.4f}")
print(f"AUC-PR:    {average_precision_score(y, proba):.4f}")
print(f"Log Loss:  {log_loss(y, proba):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y, preds))
print("\nClassification Report:")
print(classification_report(y, preds))

# Standardized coefficients
lr_model = pipe.named_steps['lr']
features = ['Age', 'Income', 'LoanAmount']
print("\nStandardized Coefficients:")
for f, c in sorted(zip(features, lr_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {f}: {c:+.4f}")

# Custom threshold comparison
print("\nThreshold Comparison:")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds_t = (proba >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score
    p = precision_score(y, preds_t, zero_division=0)
    r = recall_score(y, preds_t)
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    print(f"  Threshold={thresh}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")
```
