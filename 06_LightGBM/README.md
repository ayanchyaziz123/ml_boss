# LightGBM — Interview Questions & Answers

---

## Small Dataset Example

| Click Through Rate | Session Duration | Pages Visited | Device (0=mobile,1=desktop) | Converted? |
|--------------------|-----------------|---------------|----------------------------|------------|
| 0.05 | 120 | 4 | 1 | 1 |
| 0.02 | 30 | 1 | 0 | 0 |
| 0.08 | 200 | 7 | 1 | 1 |
| 0.01 | 15 | 1 | 0 | 0 |
| 0.10 | 250 | 9 | 1 | 1 |
| 0.03 | 45 | 2 | 0 | 0 |
| 0.07 | 180 | 6 | 0 | 1 |
| 0.01 | 20 | 1 | 0 | 0 |

### Dataset Questions

**Q: LightGBM uses leaf-wise growth. On this small dataset, what does that mean for tree shape?**
> **A:** Leaf-wise growth finds the single leaf with the highest loss reduction across the entire tree and splits it — regardless of depth. This creates **asymmetric, unbalanced trees** compared to level-wise which adds one full level at a time. On this dataset, it would quickly split on the most informative features like Session Duration.

**Q: If GOSS is applied with top_rate=0.2 and other_rate=0.1, how many samples per iteration from 8 rows?**
> **A:** Top 20% large-gradient samples = 1–2 rows (all kept). Random 10% of remaining ~6 rows = 0–1 row. GOSS focuses computation on the hardest samples. On a dataset this small, GOSS provides minimal benefit — it's designed for millions of rows.

**Q: The Device column is categorical (0/1). How does LightGBM handle it differently from XGBoost?**
> **A:** LightGBM can handle categorical features natively with `categorical_feature` parameter. It finds the optimal split by grouping categories rather than treating them as ordered numbers. XGBoost requires you to pre-encode categoricals before training.

---

## Questions & Answers

---

### Q1. What is LightGBM?

**A:** LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework developed by Microsoft. It is designed for speed and efficiency on large datasets. Key innovations:
- **GOSS** (Gradient-based One-Side Sampling) — faster data sampling
- **EFB** (Exclusive Feature Bundling) — reduces feature space
- **Leaf-wise tree growth** — faster convergence

---

### Q2. What is the difference between LightGBM and XGBoost?

**A:**
| | LightGBM | XGBoost |
|---|---|---|
| Tree growth | Leaf-wise | Level-wise |
| Speed | Much faster (especially large data) | Slower |
| Memory | Much less | More |
| Categorical features | Native support | Requires encoding |
| GOSS sampling | Yes | No |
| EFB | Yes | No |
| Overfitting risk | Higher (leaf-wise) | Moderate |
| Best for | Large datasets (>10k rows) | Medium datasets |

---

### Q3. What is leaf-wise growth?

**A:** In leaf-wise (best-first) growth, at each iteration, LightGBM:
1. Evaluates **all current leaves**
2. Splits the leaf with the **highest gain** (maximum loss reduction)
3. Only one leaf is split per iteration

This produces deeper, asymmetric trees. Faster convergence and lower loss compared to level-wise, but prone to overfitting on small datasets. Control with `num_leaves` parameter.

---

### Q4. What is level-wise growth?

**A:** Level-wise (depth-first) growth builds trees one full level at a time:
- All nodes at depth d are split before moving to depth d+1
- Produces balanced, symmetric trees
- Used by XGBoost and sklearn's GBM
- More stable on small datasets, slower convergence

**Comparison:**
- Level-wise: depth 3 = 8 leaves maximum
- Leaf-wise: depth 3 could mean only 1–3 leaves (only best splits taken)

---

### Q5. What is GOSS (Gradient-based One-Side Sampling)?

**A:** GOSS reduces training data size while preserving accuracy:
1. Keep all samples with **large gradients** (high loss) — these are the "hard" samples
2. Randomly sample a small fraction of **small-gradient** samples
3. Amplify the small-gradient samples by a constant factor to correct distribution

**Why it works:** Samples with large gradients contribute more to learning. Safe to discard most small-gradient samples.

**Parameters:** `top_rate` (fraction of large-gradient samples) and `other_rate` (fraction of small-gradient samples).

---

### Q6. What is EFB (Exclusive Feature Bundling)?

**A:** EFB reduces the number of features by bundling mutually exclusive features together:
- Two features are "exclusive" if they rarely have non-zero values simultaneously (common in one-hot encoded data)
- Bundle such features into one feature
- Reduces feature dimensionality → faster split finding

**Example:** In one-hot encoded categories, only one column is 1 at a time → all can be bundled into a single feature without information loss.

---

### Q7. Why is LightGBM faster than XGBoost?

**A:** Multiple speed improvements:
1. **GOSS** — trains on fewer samples per iteration
2. **EFB** — trains on fewer features per iteration
3. **Histogram-based algorithm** — bins continuous features into discrete bins (e.g., 255 bins), reducing split candidates
4. **Leaf-wise growth** — reaches lower loss in fewer iterations
5. **Efficient cache usage** — data stored in histograms fits in CPU cache

---

### Q8. When can LightGBM overfit?

**A:** LightGBM is more prone to overfitting than XGBoost due to leaf-wise growth:
- **Small datasets** (< 10,000 rows) — leaf-wise grows very deep trees
- **High `num_leaves`** — too many leaves memorize training data
- **Low `min_data_in_leaf`** — leaves with few samples overfit

**Prevention:**
- Reduce `num_leaves` (key parameter, more important than `max_depth`)
- Increase `min_data_in_leaf`
- Add L1/L2 regularization (`reg_alpha`, `reg_lambda`)
- Use early stopping

---

### Q9. What are important hyperparameters of LightGBM?

**A:**

| Parameter | Description | Typical Range |
|---|---|---|
| `n_estimators` | Number of trees | 100–2000 |
| `learning_rate` | Shrinkage per tree | 0.01–0.1 |
| `num_leaves` | Max leaves per tree | 20–300 |
| `max_depth` | Max depth (use with num_leaves) | -1 (no limit) to 15 |
| `min_data_in_leaf` | Min samples per leaf | 20–200 |
| `feature_fraction` | Features sampled per tree | 0.6–1.0 |
| `bagging_fraction` | Rows sampled per iteration | 0.6–1.0 |
| `bagging_freq` | Frequency of bagging | 1–10 |
| `reg_alpha` | L1 regularization | 0–5 |
| `reg_lambda` | L2 regularization | 0–5 |
| `min_split_gain` | Min gain for split | 0–1 |

---

### Q10. What are the advantages and disadvantages of LightGBM?

**A:**

**Advantages:**
1. Extremely fast on large datasets
2. Low memory usage
3. Native categorical feature support
4. High accuracy
5. Handles large number of features well
6. GPU support

**Disadvantages:**
1. Prone to overfitting on small datasets
2. Sensitive to `num_leaves` — needs careful tuning
3. Less interpretable hyperparameters than XGBoost
4. Leaf-wise growth can create very unbalanced trees
5. Less widely documented than XGBoost

---

### Q11. What is num_leaves in LightGBM?

**A:** `num_leaves` is the most important complexity parameter in LightGBM (more impactful than `max_depth`). It controls the maximum number of leaf nodes per tree.

```
num_leaves < 2^(max_depth)   — to prevent overfitting
```
- `num_leaves=31`: Default, good for most cases
- `num_leaves=127`: More complex model, needs regularization
- Rule: Set `num_leaves = 0.6 × 2^(max_depth)` as starting point

---

### Q12. What is min_data_in_leaf?

**A:** The minimum number of samples required in each leaf node. Higher values prevent overfitting but may cause underfitting.

- Small datasets (< 10k rows): `min_data_in_leaf=20–100`
- Large datasets (> 100k rows): `min_data_in_leaf=100–500`

This is the most effective regularization parameter in LightGBM alongside `num_leaves`.

---

### Q13. How does LightGBM handle categorical features?

**A:** Pass column indices to `categorical_feature` parameter. LightGBM uses a special algorithm:
1. Groups categories by their mean target value
2. Treats them as ordered for efficient split finding
3. Finds the optimal partition of categories into two groups

```python
import lightgbm as lgb
model = lgb.LGBMClassifier()
model.fit(X, y, categorical_feature=[3])  # column index 3 is categorical
```

**Requirement:** Encode categories as non-negative integers first.

---

### Q14. Why does leaf-wise growth risk overfitting?

**A:** Leaf-wise growth always splits the leaf with the maximum gain — this may create very deep branches on specific subsets of data. With enough splits, the model can memorize individual training points (deep leaf with 1 sample). Level-wise growth naturally limits this by requiring all nodes at a depth to be split equally.

**Fix:** Set `num_leaves` carefully (lower = less overfitting).

---

### Q15. How do you prevent overfitting in LightGBM?

**A:**
1. **Reduce `num_leaves`** (most important)
2. **Increase `min_data_in_leaf`**
3. **Use `feature_fraction` < 1** (column subsampling)
4. **Use `bagging_fraction` + `bagging_freq`** (row subsampling)
5. **Add `reg_alpha` and `reg_lambda`**
6. **Lower `learning_rate`** + increase `n_estimators` with early stopping
7. **Use `min_split_gain`** for minimum gain threshold

---

### Q16. What is the difference between LightGBM and CatBoost?

**A:**
| | LightGBM | CatBoost |
|---|---|---|
| Speed (training) | Very fast | Slower |
| Speed (prediction) | Fast | Very fast |
| Categorical handling | Requires integer encoding | True native (strings OK) |
| Overfitting | More prone | Less prone (ordered boosting) |
| Tree structure | Asymmetric | Symmetric (oblivious trees) |
| GPU training | Yes | Yes (very optimized) |
| Best for | Large datasets, speed | High-cardinality categoricals |

---

### Q17. How does LightGBM handle large datasets?

**A:** LightGBM is specifically designed for large-scale data:
- **Histogram binning:** Continuous features → 255 bins, dramatically reduces memory and computation
- **GOSS:** Reduces rows processed per iteration
- **EFB:** Reduces features processed per iteration
- **Memory-efficient:** Stores histograms instead of raw data
- **Multi-threading:** Parallel histogram building

---

### Q18. What is bagging_fraction and bagging_freq?

**A:**
- `bagging_fraction` (0–1): Fraction of rows randomly sampled per iteration without replacement
- `bagging_freq` (integer): Perform bagging every k iterations (0 = disable)

Similar to `subsample` in XGBoost. Both must be set together:
```python
lgb.LGBMClassifier(bagging_fraction=0.8, bagging_freq=5)
```

---

### Q19. What is feature_fraction?

**A:** `feature_fraction` (0–1) controls the fraction of features randomly selected per tree. Similar to `colsample_bytree` in XGBoost. Values < 1 add randomness, speed up training, and reduce overfitting.

---

### Q20. What evaluation metrics does LightGBM support?

**A:**
- Binary: `binary_logloss`, `binary_error`, `auc`, `average_precision`
- Multiclass: `multi_logloss`, `multi_error`
- Regression: `l1`, `l2`, `rmse`, `mape`, `huber`
- Ranking: `ndcg`, `map`

---

## Quick Code Example

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X = np.array([[0.05,120,4,1],[0.02,30,1,0],[0.08,200,7,1],[0.01,15,1,0],
              [0.10,250,9,1],[0.03,45,2,0],[0.07,180,6,0],[0.01,20,1,0]])
y = np.array([1,0,1,0,1,0,1,0])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=15,
    min_data_in_leaf=1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

print("Best Iteration:", model.best_iteration_)
print("Accuracy:", accuracy_score(y_val, model.predict(X_val)))
print("Feature Importances:", model.feature_importances_)
```
