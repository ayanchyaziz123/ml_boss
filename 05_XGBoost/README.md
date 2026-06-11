# XGBoost — Interview Questions & Answers

---

## Small Dataset Example

| Study Hours | Sleep Hours | Past Score | Attendance % | Exam Pass? |
|-------------|-------------|------------|--------------|------------|
| 8 | 7 | 75 | 90 | 1 |
| 3 | 5 | 50 | 60 | 0 |
| 6 | 8 | 68 | 85 | 1 |
| 2 | 4 | 45 | 55 | 0 |
| 9 | 7 | 82 | 95 | 1 |
| 4 | 6 | 58 | 70 | 0 |
| 7 | 8 | 72 | 88 | 1 |
| 1 | 4 | 40 | 50 | 0 |

### Dataset Questions

**Q: XGBoost trains sequentially. What does Tree 1 try to predict after Tree 0 makes its initial predictions?**
> **A:** Tree 1 tries to predict the **residual errors** (gradients) that Tree 0 made. Each subsequent tree focuses on correcting the mistakes of all previous trees combined.

**Q: If learning_rate=0.1 and a tree predicts +5 for a sample, how much is actually added to the prediction?**
> **A:** Only `0.1 × 5 = 0.5` is added to the running prediction. The learning rate shrinks each tree's contribution, preventing any single tree from dominating.

**Q: You notice the model is overfitting. Which XGBoost hyperparameters would you tune first?**
> **A:** Lower `max_depth` (e.g., 3–6), increase `min_child_weight`, add `subsample` (0.8), add `colsample_bytree` (0.8), increase `lambda` (L2 regularization), and use early stopping with a validation set.

---

## Questions & Answers

---

### Q1. What is boosting?

**A:** Boosting is an ensemble technique that builds models **sequentially**, where each new model focuses on correcting the errors of the previous models.

Key idea: **Weak learners (slightly better than random) + sequential correction = strong learner**

Unlike bagging (parallel), boosting is sequential and adaptive. Each iteration reweights the data to emphasize previously misclassified samples.

---

### Q2. What is XGBoost?

**A:** XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting developed by Chen & Guestrin (2016). It extends gradient boosting with:
- **Regularization** (L1 + L2 on leaf weights)
- **Second-order Taylor expansion** of the loss function
- **Parallel tree construction** (within each tree)
- **Built-in cross-validation** and early stopping
- **Sparsity-aware** split finding for missing values
- **Cache-aware** computation for speed

---

### Q3. Why is XGBoost so popular?

**A:**
1. **High accuracy** — consistently wins Kaggle competitions
2. **Speed** — parallel processing, cache optimization
3. **Regularization** — built-in L1/L2 prevents overfitting
4. **Missing value handling** — learns best direction for missing values
5. **Early stopping** — stops training when validation score stops improving
6. **Flexible** — supports classification, regression, ranking, custom objectives
7. **Cross-platform** — Python, R, Java, Spark

---

### Q4. What is gradient boosting?

**A:** Gradient boosting frames boosting as a gradient descent problem in **function space**:
1. Initialize with a simple prediction (e.g., mean of target)
2. Compute **pseudo-residuals** (negative gradient of loss)
3. Fit a tree to those pseudo-residuals
4. Update prediction: `F(x) = F(x) + η × tree(x)`
5. Repeat for `n_estimators` iterations

The "gradient" refers to the gradient of the loss function, not the input space.

---

### Q5. What is regularization in XGBoost?

**A:** XGBoost objective = Loss + Regularization:
```
Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₜ)

Where: Ω(f) = γT + (λ/2) Σwⱼ²  + α Σ|wⱼ|
```
- `T` = number of leaves, `γ` = minimum gain to make a split (pruning)
- `λ` = L2 regularization on leaf weights (default = 1)
- `α` = L1 regularization on leaf weights (default = 0)

This makes XGBoost more resistant to overfitting than vanilla gradient boosting.

---

### Q6. What is the learning rate (eta) in XGBoost?

**A:** The learning rate (`eta`, default = 0.3) scales the contribution of each tree:
```
F(x) = F(x) + η × tree(x)
```
- **Small learning rate** (e.g., 0.01): More trees needed, better generalization, slower training
- **Large learning rate** (e.g., 0.3): Fewer trees, faster but may overfit

Rule of thumb: **Lower learning rate + more trees = better model**. Use `eta=0.01–0.1` with `n_estimators=500–2000` and early stopping.

---

### Q7. What is early stopping?

**A:** Early stopping monitors performance on a validation set and stops training when it doesn't improve for a set number of rounds (`early_stopping_rounds`).

```python
model = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=50)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=False)
print("Best iteration:", model.best_iteration)
```

**Benefit:** Prevents overfitting without manually tuning `n_estimators`.

---

### Q8. How does XGBoost handle missing values?

**A:** XGBoost learns the **optimal default direction** for missing values during training. For each feature with missing values:
- It tries sending missing values to both left and right child
- Keeps the direction that reduces the loss more
- At prediction time, uses this learned direction for missing values

This is automatic — no imputation needed.

---

### Q9. What is the difference between XGBoost and Random Forest?

**A:**
| | XGBoost | Random Forest |
|---|---|---|
| Method | Boosting (sequential) | Bagging (parallel) |
| Corrects errors | Yes (each tree fixes previous) | No (independent trees) |
| Bias-variance | Reduces bias | Reduces variance |
| Overfitting risk | Higher (needs tuning) | Lower |
| Performance | Usually higher | Good baseline |
| Training speed | Slower | Faster |
| Hyperparameter sensitivity | High | Low |

---

### Q10. What are the important hyperparameters of XGBoost?

**A:**

| Parameter | Description | Typical Range |
|---|---|---|
| `n_estimators` | Number of trees | 100–2000 |
| `learning_rate` | Shrinkage per tree | 0.01–0.3 |
| `max_depth` | Max tree depth | 3–10 |
| `min_child_weight` | Min sum of hessian in child | 1–10 |
| `subsample` | Row sampling per tree | 0.6–1.0 |
| `colsample_bytree` | Feature sampling per tree | 0.5–1.0 |
| `gamma` | Min loss reduction for split | 0–5 |
| `lambda` | L2 regularization | 0–10 |
| `alpha` | L1 regularization | 0–10 |
| `scale_pos_weight` | Balance for imbalanced data | `neg/pos` |

---

### Q11. What is a weak learner?

**A:** A weak learner is a model that performs only slightly better than random guessing (accuracy > 50% for binary classification). In gradient boosting, shallow decision trees (depth 3–6) are typically used as weak learners.

The power of boosting: combining many weak learners sequentially creates a strong learner.

---

### Q12. What is subsample in XGBoost?

**A:** `subsample` controls the fraction of training rows used per tree (stochastic gradient boosting):
- `subsample=0.8` → each tree randomly uses 80% of training data
- Helps prevent overfitting by introducing randomness
- Also speeds up training

---

### Q13. What is colsample_bytree?

**A:** `colsample_bytree` controls the fraction of features (columns) sampled per tree:
- `colsample_bytree=0.8` → each tree randomly uses 80% of features
- Similar to the feature randomness in Random Forest
- Reduces correlation between trees

Also: `colsample_bylevel` (per tree level) and `colsample_bynode` (per split).

---

### Q14. What is max_depth in XGBoost?

**A:** `max_depth` limits how deep each tree can grow. Shallower trees = weaker learners = less overfitting.

- `max_depth=3`: Very conservative, may underfit
- `max_depth=6`: Good default for most problems
- `max_depth=10+`: Risk of overfitting

Unlike Random Forest, XGBoost is more sensitive to `max_depth` due to sequential amplification.

---

### Q15. What is the difference between XGBoost and GBM (sklearn's GradientBoostingClassifier)?

**A:**
| | XGBoost | sklearn GBM |
|---|---|---|
| Speed | Much faster | Slower |
| Regularization | L1 + L2 | None |
| Missing values | Handles natively | No |
| Parallel within tree | Yes | No |
| Second-order gradients | Yes | No |
| Early stopping | Built-in | Manual |

---

### Q16. What is tree pruning in XGBoost?

**A:** XGBoost uses a different pruning strategy — it builds trees up to `max_depth`, then **prunes backwards**. A split is removed if its gain is less than `gamma` (minimum gain threshold).

This "max-depth first, then prune" approach differs from the greedy early stopping in sklearn trees.

---

### Q17. What is gamma (min_split_loss) in XGBoost?

**A:** `gamma` is the minimum reduction in loss required to make a split. Higher gamma = more conservative, fewer splits, simpler trees.

- `gamma=0`: No restriction (default)
- `gamma=5`: Only make splits that reduce loss by at least 5

Useful for controlling tree complexity.

---

### Q18. What is the difference between XGBoost and LightGBM?

**A:**
| | XGBoost | LightGBM |
|---|---|---|
| Growth strategy | Level-wise (depth-first) | Leaf-wise |
| Speed | Slower on large data | Much faster |
| Memory | Higher | Lower (GOSS + EFB) |
| Accuracy | Similar | Similar or slightly better |
| Overfitting risk | Moderate | Higher (leaf-wise needs `num_leaves` control) |
| Categorical features | Requires encoding | Native support |

---

### Q19. What is the difference between XGBoost and CatBoost?

**A:**
| | XGBoost | CatBoost |
|---|---|---|
| Categorical features | Requires encoding | Native (ordered target encoding) |
| Target leakage | Can occur with encoding | Prevents via ordered boosting |
| Speed | Fast | Slower training, fast prediction |
| GPU support | Yes | Yes (very optimized) |
| Symmetric trees | No | Yes (oblivious trees) |

---

### Q20. How does the second-order Taylor expansion improve XGBoost?

**A:** XGBoost approximates the loss using a second-order Taylor expansion:
```
L ≈ Σ [gᵢf(xᵢ) + (1/2)hᵢf(xᵢ)²]
```
Where `gᵢ` = first derivative (gradient) and `hᵢ` = second derivative (hessian).

Using both gradient and curvature information allows:
- More precise leaf weight calculation: `w* = -G/H`
- Better split scoring
- Works with any differentiable loss function

---

## Quick Code Example

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

X = np.array([[8,7,75,90],[3,5,50,60],[6,8,68,85],[2,4,45,55],
              [9,7,82,95],[4,6,58,70],[7,8,72,88],[1,4,40,50]])
y = np.array([1,0,1,0,1,0,1,0])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=False)

print("Best Iteration:", model.best_iteration)
print("Accuracy:", accuracy_score(y_val, model.predict(X_val)))
print(classification_report(y_val, model.predict(X_val)))
```
