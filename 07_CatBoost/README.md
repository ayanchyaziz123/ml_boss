# CatBoost — Interview Questions & Answers

---

## Small Dataset Example

| City | Job Type | Education | Experience (yrs) | Salary > 50k? |
|------|----------|-----------|-----------------|---------------|
| NY | Engineer | Masters | 5 | 1 |
| LA | Sales | Bachelors | 2 | 0 |
| NY | Manager | Masters | 8 | 1 |
| Chicago | Clerk | Bachelors | 1 | 0 |
| SF | Engineer | PhD | 10 | 1 |
| LA | Clerk | HighSchool | 3 | 0 |
| NY | Manager | Bachelors | 6 | 1 |
| Chicago | Sales | Bachelors | 2 | 0 |

### Dataset Questions

**Q: This dataset has 3 categorical features (City, Job Type, Education). How does CatBoost process them without manual encoding?**
> **A:** CatBoost uses **Ordered Target Encoding** with random permutations. For each categorical value, it computes the target mean using only the samples seen *before* the current sample in the permutation — this prevents target leakage. No manual Label Encoding or One-Hot Encoding is needed; just pass column names to `cat_features`.

**Q: What is the risk if you used regular target encoding on this dataset?**
> **A:** Regular target encoding replaces each category with the mean target across ALL rows — including the current row. This leaks the target into the features, causing the model to memorize training data and fail to generalize (overfitting and overly optimistic validation scores).

**Q: CatBoost builds symmetric trees. What does that mean for this dataset?**
> **A:** Each level of the tree uses the **same feature and threshold** for all nodes at that level. For example, if depth 1 splits on Experience > 4, then ALL nodes at depth 1 use Experience > 4. This makes prediction fast (lookup table) and reduces overfitting.

---

## Questions & Answers

---

### Q1. What is CatBoost?

**A:** CatBoost (Categorical Boosting) is a gradient boosting library developed by Yandex in 2017. It is specifically designed to handle categorical features natively and prevent overfitting through **ordered boosting**. Key features:
- Native categorical feature support (strings, integers)
- Ordered (permutation-based) boosting
- Symmetric (oblivious) trees
- Strong default parameters — minimal tuning needed
- Fast GPU training and prediction

---

### Q2. How does CatBoost handle categorical features?

**A:** CatBoost uses **Ordered Target Statistics** (target encoding with protection against leakage):

For each training sample, the target encoding for a category is computed using only the samples that appear **before** it in a random permutation:
```
encoding(xᵢ) = (sum of target in previous samples with same category + prior) /
               (count of previous samples with same category + 1)
```
Multiple random permutations are used to get robust estimates.

This means: no manual one-hot encoding, label encoding, or frequency encoding needed.

---

### Q3. What is ordered boosting?

**A:** Standard gradient boosting has a subtle problem: the residuals (gradients) for each sample are computed using a model trained on the **same sample** → leads to overfitting. 

**Ordered boosting** fixes this:
1. Randomly permute the training data
2. For sample i, compute its gradient using a model trained only on samples 1 to i-1
3. Each sample gets its gradient from a model that never saw it

This is similar to out-of-fold estimation and provides built-in regularization.

---

### Q4. What is target leakage and how does CatBoost prevent it?

**A:** Target leakage occurs when information about the target variable is inadvertently included in the features during training — the model sees the answer it's trying to predict.

**In target encoding:** If you encode `City = NY` as `mean(salary for NY rows)`, and the current NY row is included in that mean, information about its target leaks into the feature.

**CatBoost's prevention:** Use only previous samples (in a random permutation) to compute each sample's encoding → each sample's encoding is computed without knowing its own target.

---

### Q5. What are symmetric trees in CatBoost?

**A:** CatBoost builds **oblivious (symmetric) decision trees** where every node at the same depth uses the **same feature and threshold**:

```
Level 0: root
Level 1: left child (Experience ≤ 4), right child (Experience > 4)
Level 2: all nodes split on: City = NY  (same split everywhere)
Level 3: all nodes split on: Education = Masters (same split everywhere)
```

**Benefits:**
- Much faster prediction — implemented as a lookup table
- Natural regularization — prevents individual branches from overfitting
- Good for GPU parallelization

**Downside:** Less expressive than asymmetric trees for complex patterns.

---

### Q6. How is CatBoost different from XGBoost?

**A:**
| | CatBoost | XGBoost |
|---|---|---|
| Categorical features | Native (no encoding needed) | Manual encoding required |
| Tree structure | Symmetric (oblivious) | Asymmetric |
| Ordered boosting | Yes (prevents leakage) | No |
| Default tuning | Excellent defaults | Needs more tuning |
| Training speed | Slower | Faster |
| Prediction speed | Very fast | Fast |
| GPU efficiency | Very optimized | Good |
| Best for | High-cardinality categorical data | Numerical data, competitions |

---

### Q7. How is CatBoost different from LightGBM?

**A:**
| | CatBoost | LightGBM |
|---|---|---|
| Categorical features | True native (string input) | Needs integer encoding |
| Tree growth | Level-wise (symmetric) | Leaf-wise |
| Overfitting | Less prone | More prone (small datasets) |
| Training speed | Slower | Faster |
| Ordered boosting | Yes | No |
| Best for | Data with many categoricals | Large numerical datasets |

---

### Q8. What are the key hyperparameters of CatBoost?

**A:**

| Parameter | Description | Typical Range |
|---|---|---|
| `iterations` | Number of trees | 100–2000 |
| `learning_rate` | Shrinkage | 0.01–0.1 |
| `depth` | Tree depth (symmetric) | 4–10 |
| `l2_leaf_reg` | L2 regularization | 1–10 |
| `border_count` | Bins for numerical features | 32–255 |
| `bagging_temperature` | Controls bootstrap randomness | 0–1 |
| `random_strength` | Score noise for regularization | 0–10 |
| `od_type` | Overfitting detector type | `Iter`, `IncToDec` |
| `od_wait` | Early stopping patience | 20–100 |

---

### Q9. What is the depth parameter in CatBoost?

**A:** `depth` controls the depth of symmetric trees. Since CatBoost uses oblivious trees, a depth-6 tree has `2^6 = 64` leaves (all paths from root to leaf have the same length).

- `depth=4`: Simple, fast, may underfit — good for large datasets
- `depth=6`: Default, good balance
- `depth=10`: Complex, slower, more prone to overfitting

**Note:** CatBoost `depth=6` is more constrained than XGBoost `max_depth=6` due to symmetric structure.

---

### Q10. When should you choose CatBoost over XGBoost?

**A:** Choose CatBoost when:
1. **Many categorical features** with high cardinality (cities, product IDs, user types)
2. **You want minimal preprocessing** — no encoding pipeline needed
3. **You need strong out-of-box performance** — CatBoost defaults are excellent
4. **Target leakage risk** with target encoding in other methods
5. **Fast inference** is required (symmetric tree lookup tables)

Choose XGBoost when:
- Mostly numerical features
- You need faster training
- Competing in Kaggle (more community resources)

---

### Q11. What is the `cat_features` parameter?

**A:** Pass the list of categorical feature indices or names to `cat_features`:

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=500, depth=6)
model.fit(X, y, cat_features=[0, 1, 2])  # City, Job Type, Education columns
# OR with feature names:
model.fit(X_df, y, cat_features=['City', 'JobType', 'Education'])
```

CatBoost accepts **raw string categories** — no need to encode first.

---

### Q12. Does CatBoost require feature scaling?

**A:** No. Like all tree-based models, CatBoost makes splits based on relative order of values, not absolute values. Feature scaling (normalization/standardization) has no effect on CatBoost's performance.

---

### Q13. What is bagging_temperature in CatBoost?

**A:** `bagging_temperature` controls the randomness of the Bayesian bootstrap:
- `0`: No randomness (full dataset used)
- `1`: Standard Bayesian bootstrap (default)
- Higher values: More randomness

Unlike `subsample` (which takes a fixed fraction), Bayesian bootstrap assigns random weights from an exponential distribution to training samples.

---

### Q14. What is ordered target encoding vs standard target encoding?

**A:**
| | Standard Target Encoding | Ordered Target Encoding (CatBoost) |
|---|---|---|
| Uses current sample? | Yes → leakage | No → safe |
| Training data | All rows | Only previous rows (permutation) |
| Validation reliability | Biased | Unbiased |
| Computation | Simple | More complex |
| Overfitting | Prone | Resistant |

---

### Q15. What is early stopping in CatBoost?

**A:** CatBoost has a built-in overfitting detector:

```python
model = CatBoostClassifier(
    iterations=1000,
    od_type='Iter',     # Stop if no improvement
    od_wait=50,         # Stop after 50 rounds without improvement
    eval_metric='AUC'
)
model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          verbose=100)
```

`od_type='IncToDec'` stops when the metric starts decreasing from its peak.

---

### Q16. How does CatBoost handle missing values?

**A:** CatBoost handles missing values automatically:
- **Numerical features:** Missing values are treated as a special value and placed in a separate bin during histogram building
- **Categorical features:** Missing values are treated as a special category ("NaN")

No imputation needed before training.

---

### Q17. What is the difference between CatBoost and gradient boosting conceptually?

**A:** CatBoost = Gradient Boosting + three key innovations:
1. **Ordered boosting** — prevents overfitting via permutation-based gradient computation
2. **Ordered target statistics** — prevents leakage in categorical encoding
3. **Symmetric trees** — fast prediction and natural regularization

The base algorithm (sequential tree building on negative gradients) is the same as standard gradient boosting.

---

### Q18. Can CatBoost be used for regression?

**A:** Yes. Use `CatBoostRegressor`:

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE'
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

Supported regression metrics: `RMSE`, `MAE`, `MAPE`, `Huber`, `Quantile`.

---

### Q19. How does CatBoost perform feature importance?

**A:** CatBoost provides multiple importance types:

```python
# PredictionValuesChange (default)
model.get_feature_importance()

# LossFunctionChange (more accurate)
model.get_feature_importance(type='LossFunctionChange',
                              data=Pool(X_val, y_val))

# ShapValues (SHAP-based)
shap_values = model.get_feature_importance(type='ShapValues',
                                            data=Pool(X_val, y_val))
```

---

### Q20. What is the Pool object in CatBoost?

**A:** `Pool` is CatBoost's data container that stores features, labels, and metadata together:

```python
from catboost import Pool

train_pool = Pool(X_train, y_train,
                  cat_features=['City', 'JobType', 'Education'],
                  feature_names=['City', 'JobType', 'Education', 'Exp'])

model.fit(train_pool)
```

Using Pool is optional but improves performance (especially for repeated operations) and is required for SHAP computation.

---

## Quick Code Example

```python
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data = {
    'City': ['NY','LA','NY','Chicago','SF','LA','NY','Chicago'],
    'JobType': ['Engineer','Sales','Manager','Clerk','Engineer','Clerk','Manager','Sales'],
    'Education': ['Masters','Bachelors','Masters','Bachelors','PhD','HighSchool','Bachelors','Bachelors'],
    'Experience': [5, 2, 8, 1, 10, 3, 6, 2]
}
X = pd.DataFrame(data)
y = [1, 0, 1, 0, 1, 0, 1, 0]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

model = CatBoostClassifier(
    iterations=200,
    depth=4,
    learning_rate=0.1,
    cat_features=['City', 'JobType', 'Education'],
    verbose=0
)

model.fit(X_train, y_train, eval_set=(X_val, y_val))

print("Accuracy:", accuracy_score(y_val, model.predict(X_val)))
print("Feature Importances:", model.get_feature_importance())
```
