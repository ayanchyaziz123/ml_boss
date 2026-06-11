# Feature Engineering — Interview Questions & Answers

---

## Small Dataset Example

| Age | Salary | Join Date | Department | City | Churned? |
|-----|--------|-----------|------------|------|----------|
| 28 | 55000 | 2020-03-15 | Engineering | NY | 0 |
| 45 | 92000 | 2018-07-22 | Marketing | LA | 1 |
| 32 | 67000 | 2021-01-10 | Engineering | NY | 0 |
| 55 | 110000 | 2015-11-05 | Management | SF | 0 |
| 24 | 42000 | 2022-05-20 | Sales | LA | 1 |
| 38 | 78000 | 2019-08-30 | Marketing | NY | 1 |

### Dataset Questions

**Q: How would you transform the "Join Date" column into useful features?**
> **A:** Extract: (1) **Tenure** = current_date − join_date in days/months/years (numeric, captures loyalty), (2) **Join Year** = 2020, 2018... (trend feature), (3) **Join Month** = 3, 7, 1... (seasonality), (4) **Join Quarter** = 1,3,1... (cyclical). Avoid using raw dates — most ML models can't use datetime objects directly.

**Q: The "Department" column has 4 unique values. Should you use Label Encoding or One-Hot Encoding?**
> **A:** Use **One-Hot Encoding** for tree models and neural networks (Engineering=0001, Marketing=0100, etc.) because Department has no natural ordering. Label Encoding (Engineering=0, Marketing=1, Management=2, Sales=3) would incorrectly imply Management > Marketing. Exception: tree-based models like CatBoost handle raw categoricals natively.

**Q: How would you create a new feature combining Age and Salary?**
> **A:** Create **Salary per Year of Age** = Salary / Age (career earning ratio). Or bin Age into groups and compute mean Salary per group as a target encoding. Feature interactions often capture non-linear relationships better than individual features.

---

## Questions & Answers

---

### Q1. What is feature engineering?

**A:** Feature engineering is the process of using domain knowledge to create, transform, or select features that improve model performance. It bridges raw data and the model:

- **Creation:** Make new features (Age from BirthDate, BMI from Height/Weight)
- **Transformation:** Change scale or distribution (log, normalize, bin)
- **Selection:** Keep only the most informative features
- **Encoding:** Convert categorical/text features to numbers

---

### Q2. What is feature selection?

**A:** Feature selection identifies and keeps only the most relevant features for the model. Methods:

- **Filter:** Correlation, mutual information, chi-square — rank features independently
- **Wrapper:** RFE (Recursive Feature Elimination) — iteratively remove worst features
- **Embedded:** L1/Lasso, tree feature importance — selection during model training

```python
from sklearn.feature_selection import RFE, mutual_info_classif
mi = mutual_info_classif(X, y)  # filter method
```

---

### Q3. What is feature extraction?

**A:** Feature extraction creates new, informative features from raw data:
- From images: CNN activations, HOG descriptors, color histograms
- From text: TF-IDF, BERT embeddings
- From time series: rolling mean, lag features, FFT frequency features
- From datetime: hour, day_of_week, is_weekend, days_since_event

Differs from feature selection (which picks from existing features).

---

### Q4. What is one-hot encoding?

**A:** One-hot encoding creates a binary column for each unique category value:

```
Department: [Engineering, Marketing, Sales]
Engineering → [1, 0, 0]
Marketing   → [0, 1, 0]
Sales       → [0, 0, 1]
```

```python
pd.get_dummies(df['Department'])
# or
from sklearn.preprocessing import OneHotEncoder
```

Pros: No false ordinal relationship. Cons: High-cardinality features create many columns (curse of dimensionality).

---

### Q5. What is label encoding?

**A:** Label encoding assigns a unique integer to each category:

```
Engineering → 0, Marketing → 1, Management → 2, Sales → 3
```

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Dept_encoded'] = le.fit_transform(df['Department'])
```

**Use only when:** There's a natural ordinal relationship (Low=0, Medium=1, High=2). For non-ordinal categories with non-tree models, use one-hot encoding.

---

### Q6. What is ordinal encoding?

**A:** Ordinal encoding explicitly maps categories to ordered integers based on domain knowledge:

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
enc.fit_transform([['Low'], ['High'], ['Medium']])
# → [[0.], [2.], [1.]]
```

Differs from label encoding: ordinal encoding preserves the meaningful order you specify.

---

### Q7. What is target encoding?

**A:** Target encoding replaces a category with the mean target value for that category:

```
City: NY → mean(Churned for NY rows) = 0.33
      LA → mean(Churned for LA rows) = 1.0
      SF → mean(Churned for SF rows) = 0.0
```

Powerful for high-cardinality features. **Risk:** Target leakage — using the target to encode features. Use cross-validation or CatBoost's ordered encoding.

```python
from category_encoders import TargetEncoder
enc = TargetEncoder()
X['City_encoded'] = enc.fit_transform(X['City'], y)
```

---

### Q8. What is normalization (Min-Max scaling)?

**A:** Scales features to [0, 1] range:
```
x_norm = (x − x_min) / (x_max − x_min)
```

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**When to use:** When you know the range, for neural networks, or when distribution is NOT Gaussian. **Sensitive to outliers** (outliers stretch the range).

---

### Q9. What is standardization (Z-score scaling)?

**A:** Transforms feature to have mean=0 and std=1:
```
x_std = (x − μ) / σ
```

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**When to use:** When feature distribution is approximately Gaussian, for SVMs, KNN, PCA, linear models. **Less sensitive to outliers** than min-max scaling.

---

### Q10. What is the difference between normalization and standardization?

**A:**
| | Normalization (Min-Max) | Standardization (Z-score) |
|---|---|---|
| Range | [0, 1] | No fixed range |
| Assumes distribution | No assumption | Gaussian |
| Outlier sensitivity | High | Low |
| Use case | Neural networks, NNs, KNN | SVM, PCA, LR |
| Formula | (x−min)/(max−min) | (x−μ)/σ |

---

### Q11. What is log transformation?

**A:** Log transformation reduces right-skewed distributions and handles large value ranges:
```python
X['Salary_log'] = np.log1p(X['Salary'])  # log(1+x) handles zeros
```

**Use when:** Feature has a long right tail (income, prices, population), makes distribution more Gaussian, useful for linear models.

**Effect:** Salary: [42000, 55000, 110000] → log: [10.6, 10.9, 11.6] — compressed scale.

---

### Q12. What is binning (discretization)?

**A:** Binning converts a continuous feature into discrete buckets:

```python
# Equal-width binning
pd.cut(df['Age'], bins=3, labels=['Young', 'Mid', 'Senior'])

# Equal-frequency (quantile) binning
pd.qcut(df['Salary'], q=3, labels=['Low', 'Mid', 'High'])
```

**Use when:** Non-linear relationship with target, or to add regularization. **Caveat:** Loses information within bins.

---

### Q13. What is imputation?

**A:** Imputation fills in missing values:

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Mean imputation
imp = SimpleImputer(strategy='mean')
# Median imputation (robust to outliers)
imp = SimpleImputer(strategy='median')
# Mode imputation (categorical)
imp = SimpleImputer(strategy='most_frequent')
# KNN imputation (uses neighbor values)
imp = KNNImputer(n_neighbors=5)
```

**Always add an indicator column:** `X['Age_missing'] = X['Age'].isna().astype(int)` — missingness itself may be informative.

---

### Q14. What is polynomial feature creation?

**A:** Create interaction and polynomial terms to capture non-linear relationships:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False)
X_poly = poly.fit_transform(X)
# Creates: x₁, x₂, x₁², x₁x₂, x₂²
```

Useful for linear models to learn non-linear boundaries. Risk: feature explosion with high degree or many inputs.

---

### Q15. What is feature interaction?

**A:** Feature interactions capture the combined effect of two or more features that is not captured by either alone:

```python
# Manual interaction
df['Age_x_Salary'] = df['Age'] * df['Salary']
df['Salary_per_Year'] = df['Salary'] / df['Age']

# Ratio features
df['Debt_Ratio'] = df['Debt'] / df['Income']
```

Domain knowledge helps identify meaningful interactions. Tree models capture interactions automatically.

---

### Q16. What is feature importance?

**A:** Feature importance measures how much each feature contributes to model predictions:

```python
# Tree-based importance (MDI)
model.feature_importances_

# Permutation importance (model-agnostic, more reliable)
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val, n_repeats=10)

# SHAP values (best - shows direction and magnitude)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

---

### Q17. What is data leakage?

**A:** Data leakage occurs when information from outside the training period/scope leaks into features — model performs well in validation but fails in production.

**Types:**
1. **Target leakage:** Feature computed using the target variable
2. **Train-test leakage:** Fitting scaler/encoder on entire dataset (including test set)
3. **Temporal leakage:** Using future data to predict past events

**Fix:** Always fit preprocessing only on training data; use temporal cross-validation for time series.

---

### Q18. What is correlation analysis in feature selection?

**A:** Features highly correlated with the target are useful; features highly correlated with each other are redundant:

```python
# Feature-target correlation
corr = df.corr()['target'].abs().sort_values(ascending=False)

# Feature-feature correlation (remove redundant)
corr_matrix = df.drop('target', axis=1).corr()
# Remove features where |corr| > 0.95 with another feature

# Mutual information (handles non-linear relationships)
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X, y)
```

---

### Q19. What is the difference between feature selection and regularization?

**A:**
| | Feature Selection | Regularization (L1) |
|---|---|---|
| When | Before/after training | During training |
| Method | Explicit removal | Implicit shrinkage |
| Interpretability | Clear — feature removed | Coefficient → 0 |
| Flexibility | Any model | Model must support it |
| Computational cost | Can be expensive (RFE) | Just adds penalty term |

Lasso (L1) is essentially feature selection during training.

---

### Q20. What is cyclical encoding?

**A:** For cyclic features (hour 0–23, month 1–12, day of week 0–6), using raw numbers implies 23 is far from 0, but in reality they're adjacent (11pm and midnight are close). Fix: encode as sine and cosine:

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

Now hour=0 and hour=23 are close in the (sin, cos) space — correctly capturing cyclicality.

---

## Quick Code Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Dataset
data = {
    'Age': [28, 45, 32, 55, 24, 38],
    'Salary': [55000, 92000, 67000, 110000, 42000, 78000],
    'Department': ['Engineering','Marketing','Engineering','Management','Sales','Marketing'],
    'City': ['NY','LA','NY','SF','LA','NY'],
    'Churned': [0, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)
X = df.drop('Churned', axis=1)
y = df['Churned']

# Feature engineering
X['Salary_log'] = np.log1p(X['Salary'])
X['Age_group'] = pd.cut(X['Age'], bins=[0,30,45,100], labels=['Young','Mid','Senior'])

# Column transformer
numeric_features = ['Age', 'Salary', 'Salary_log']
categorical_features = ['Department', 'City', 'Age_group']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

X_processed = preprocessor.fit_transform(X)
print("Original shape:", X[numeric_features].shape)
print("Processed shape:", X_processed.shape)

# Feature names
ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_names = numeric_features + list(ohe_names)
print("\nFeature names:", all_names)
```
