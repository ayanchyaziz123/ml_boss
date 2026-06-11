# Random Forest — Interview Questions & Answers

---

## Small Dataset Example

| Age | Income | Credit Score | Debt Ratio | Loan Approved? |
|-----|--------|-------------|------------|----------------|
| 30 | 55000 | 720 | 0.3 | 1 |
| 22 | 28000 | 580 | 0.6 | 0 |
| 45 | 92000 | 760 | 0.2 | 1 |
| 35 | 47000 | 640 | 0.5 | 0 |
| 50 | 110000 | 800 | 0.1 | 1 |
| 28 | 35000 | 610 | 0.55 | 0 |
| 38 | 72000 | 700 | 0.35 | 1 |
| 26 | 31000 | 590 | 0.65 | 0 |

### Dataset Questions

**Q: Random Forest trains 100 trees on this dataset. How does it make the final prediction for a new applicant?**
> **A:** Each of the 100 trees independently predicts 0 or 1. The final prediction is decided by **majority voting** — whichever class gets more votes from the 100 trees is the final prediction. For probabilities, it averages the class probabilities across all trees.

**Q: In bootstrap sampling, if we have 8 rows, how many rows does each tree train on?**
> **A:** Each tree trains on 8 rows sampled **with replacement** from the original 8 rows. This means some rows appear multiple times and others not at all (~37% of original samples are excluded on average — those become the OOB samples).

**Q: Which feature do you expect to have the highest importance?**
> **A:** Credit Score or Debt Ratio, as they are typical strong predictors of loan approval. Feature importance is calculated as the total reduction in impurity (Gini) contributed by each feature across all trees.

---

## Questions & Answers

---

### Q1. What is Random Forest?

**A:** Random Forest is an ensemble learning algorithm that builds multiple Decision Trees and combines their predictions. It uses two randomization techniques:
1. **Bootstrap sampling** — each tree trains on a random sample of data (with replacement)
2. **Feature randomness** — each split considers only a random subset of features

Final prediction: **majority vote** (classification) or **average** (regression).

---

### Q2. What is bagging?

**A:** Bagging (Bootstrap Aggregating) is an ensemble technique that:
1. Creates multiple bootstrap samples from the training data
2. Trains a base model on each sample independently
3. Combines predictions by voting or averaging

**Goal:** Reduce variance without increasing bias. Each model is trained independently (unlike boosting).

---

### Q3. What is bootstrap sampling?

**A:** Bootstrap sampling creates a new dataset by sampling **n rows with replacement** from the original n-row dataset. Each bootstrap sample:
- Has the same size as the original dataset
- Contains some duplicates (~63.2% unique rows)
- Leaves out ~36.8% of original rows (these become OOB samples)

---

### Q4. What is OOB error?

**A:** OOB (Out-of-Bag) error is a built-in cross-validation estimate. For each training sample, it is predicted only by trees that did **not** train on it (since ~37% of samples are left out per tree). The OOB error averages these predictions across all samples.

**Advantage:** Free validation without needing a separate validation set.

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X, y)
print("OOB Score:", rf.oob_score_)
```

---

### Q5. How does Random Forest reduce overfitting?

**A:** Three mechanisms:
1. **Bootstrap sampling** — each tree sees slightly different data, reducing correlation between trees
2. **Feature randomness** — prevents all trees from using the same dominant features, decorrelating them
3. **Averaging/voting** — averages out individual tree errors, reducing variance

The key insight: individual trees overfit but in **different ways** — averaging them cancels out the overfitting.

---

### Q6. How is feature importance computed in Random Forest?

**A:** Feature importance = mean decrease in Gini impurity (or information gain) contributed by that feature across all trees and all nodes.

```
Importance(feature) = Σ (over all trees, all nodes where feature is used)
                        [impurity decrease × proportion of samples]
```

Normalized to sum to 1. Higher = more important.

```python
importances = rf.feature_importances_
for feat, imp in zip(feature_names, importances):
    print(f"{feat}: {imp:.4f}")
```

**Caveat:** This measure is biased toward high-cardinality features. Use permutation importance for unbiased estimates.

---

### Q7. What are the key hyperparameters of Random Forest?

**A:**

| Hyperparameter | Description | Typical Values |
|---|---|---|
| `n_estimators` | Number of trees | 100–500 |
| `max_depth` | Max tree depth | None, 10–30 |
| `max_features` | Features per split | `sqrt(p)` for classification, `p/3` for regression |
| `min_samples_split` | Min samples to split | 2–20 |
| `min_samples_leaf` | Min samples in leaf | 1–10 |
| `max_samples` | Bootstrap sample size | 0.8 |
| `oob_score` | Use OOB for validation | True/False |

---

### Q8. What are the advantages of Random Forest over a single Decision Tree?

**A:**
1. **Much lower variance** — averaging multiple trees reduces overfitting
2. **More accurate** — ensemble consistently outperforms individual trees
3. **Built-in OOB validation** — no need for a separate validation set
4. **Feature importance** — measures relative importance of each feature
5. **Robust to outliers** — majority voting dilutes impact of outliers
6. **Handles missing values** — via surrogate splits

---

### Q9. What is the difference between Random Forest and XGBoost?

**A:**
| | Random Forest | XGBoost |
|---|---|---|
| Type | Bagging | Boosting |
| Tree building | Parallel (independent) | Sequential (corrects errors) |
| Bias-Variance | Low variance focus | Low bias focus |
| Speed | Faster to train | Slower (sequential) |
| Overfitting | Less prone | More prone (needs tuning) |
| Performance | Good baseline | Often higher accuracy |
| Interpretability | Moderate | Moderate |

---

### Q10. When should you use Random Forest?

**A:** Use Random Forest when:
- You want a strong baseline quickly with minimal tuning
- You have high-dimensional data with many features
- You need built-in feature importance
- Data has outliers (robust to them)
- Training time is a constraint (parallelizable)

Prefer XGBoost/LightGBM when you need maximum accuracy and have time to tune.

---

### Q11. What is the difference between Random Forest and Bagging?

**A:** Random Forest is a special case of Bagging:
- **Bagging** uses full feature set at each split
- **Random Forest** uses a random subset of features at each split (`max_features=sqrt(p)`)

Feature randomness in Random Forest further decorrelates trees, typically giving better performance than plain bagging with Decision Trees.

---

### Q12. How does Random Forest make a final prediction?

**A:**
- **Classification:** Each tree votes for a class → final class = majority vote
- **Classification (probability):** Each tree outputs class probabilities → final probability = average across all trees
- **Regression:** Each tree predicts a number → final output = mean of all tree predictions

---

### Q13. Can Random Forest be used for regression?

**A:** Yes. `RandomForestRegressor` in sklearn. Each tree predicts a continuous value, and the final prediction is the **mean** of all tree predictions. Split criterion is MSE (Mean Squared Error) instead of Gini.

---

### Q14. What is the role of randomness in Random Forest?

**A:** Two sources of randomness:
1. **Data randomness (Bootstrap):** Each tree gets a different random subset of training data
2. **Feature randomness:** At each split, only `max_features` randomly selected features are considered

Together, they create diverse trees that make different errors, which cancel out when averaged.

---

### Q15. What is the difference between Random Forest and Extra Trees (Extremely Randomized Trees)?

**A:**
| | Random Forest | Extra Trees |
|---|---|---|
| Bootstrap | Yes (sampled with replacement) | No (uses full dataset) |
| Split threshold | Best of `max_features` | **Completely random** threshold |
| Speed | Slower | Faster (no optimization of threshold) |
| Variance | Moderate | Lower |
| Bias | Lower | Slightly higher |

Extra Trees adds more randomness at splits → more variance reduction but slightly higher bias.

---

### Q16. Is Random Forest sensitive to outliers?

**A:** Relatively robust compared to Linear Regression. Reasons:
- Majority voting dilutes influence of individual wrong predictions
- Decision trees split on threshold values, not exact values
- Bootstrap sampling may exclude some outliers from some trees

However, extreme outliers can still affect feature importance scores.

---

### Q17. How does Random Forest handle high-dimensional sparse data?

**A:** Random Forest works reasonably well with high-dimensional data because:
- Feature randomness at each split explores many feature combinations
- Trees can capture non-linear relationships

However, for very sparse data (like text with 10,000+ features), linear models or gradient boosting with specific tuning often outperform Random Forest.

---

### Q18. What is n_estimators and how does it affect the model?

**A:** `n_estimators` = number of trees in the forest.

- More trees → lower variance, more stable predictions
- Adding trees always helps or has no effect (never hurts accuracy)
- Diminishing returns after ~100–200 trees
- Tradeoff: more trees = slower training and prediction

Rule of thumb: start with 100, increase if OOB error is still improving.

---

### Q19. How does Random Forest compare to a single Decision Tree?

**A:** Random Forest almost always wins:
- Decision Tree: High variance, sensitive to small data changes
- Random Forest: Low variance, stable predictions

The cost is: Random Forest is less interpretable (can't visualize 100 trees) and slower. For interpretability, use a single Decision Tree.

---

### Q20. What is permutation importance and how does it differ from built-in importance?

**A:**
- **Built-in (MDI) importance:** Based on Gini/impurity reduction during training. Biased toward high-cardinality numerical features.
- **Permutation importance:** Randomly shuffle a feature's values → measure accuracy drop. More reliable and unbiased.

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10)
print(result.importances_mean)
```

---

## Quick Code Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.array([[30,55000,720,0.3],[22,28000,580,0.6],[45,92000,760,0.2],
              [35,47000,640,0.5],[50,110000,800,0.1],[28,35000,610,0.55],
              [38,72000,700,0.35],[26,31000,590,0.65]])
y = np.array([1,0,1,0,1,0,1,0])

rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                             max_features='sqrt', oob_score=True,
                             random_state=42)
rf.fit(X, y)

print("OOB Score:", rf.oob_score_)
print("CV Score:", cross_val_score(rf, X, y, cv=3).mean())
print("Feature Importances:", rf.feature_importances_)
```
