# Linear Regression — Interview Questions & Answers

---

## Small Dataset Example

| House Size (sqft) | Bedrooms | Age (years) | Price ($1000) |
|-------------------|----------|-------------|----------------|
| 1400 | 3 | 10 | 245 |
| 1600 | 3 | 5 | 312 |
| 1700 | 4 | 8 | 279 |
| 1875 | 4 | 3 | 308 |
| 1100 | 2 | 15 | 199 |
| 1550 | 3 | 7 | 219 |
| 2350 | 4 | 2 | 405 |
| 2450 | 4 | 1 | 324 |

### Dataset Questions

**Q: Using the dataset above, which feature do you think has the strongest linear relationship with Price?**
> **A:** House Size (sqft) likely has the strongest positive correlation with Price, as larger homes tend to cost more. You would verify this by computing the Pearson correlation coefficient between each feature and Price.

**Q: If you fit a Linear Regression on this dataset, what would be the interpretation of the coefficient for Bedrooms?**
> **A:** The coefficient for Bedrooms represents the average change in Price (in $1000s) for each additional bedroom, holding all other features constant.

**Q: House with 1000 sqft, 2 bedrooms, age 20 years — the model predicts $150k but actual is $180k. What is the residual?**
> **A:** Residual = Actual − Predicted = 180 − 150 = **$30k**. A positive residual means the model underestimated.

---

## Questions & Answers

---

### Q1. What is Linear Regression?

**A:** Linear Regression is a supervised learning algorithm that models the relationship between a continuous dependent variable (target) and one or more independent variables (features) by fitting a straight line.

The equation is:
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```
- `β₀` = intercept (bias)
- `β₁...βₙ` = coefficients (weights)
- `ŷ` = predicted value

---

### Q2. What assumptions does Linear Regression make?

**A:** Five key assumptions:
1. **Linearity** — relationship between features and target is linear
2. **Independence** — observations are independent of each other
3. **Homoscedasticity** — constant variance of residuals across all values
4. **Normality of residuals** — residuals are normally distributed
5. **No multicollinearity** — features are not highly correlated with each other

---

### Q3. What is the cost function in Linear Regression?

**A:** Mean Squared Error (MSE):
```
MSE = (1/n) Σ(yᵢ − ŷᵢ)²
```
We minimize this to find the best-fit line. MSE penalizes large errors more than small ones because of the squaring.

---

### Q4. Explain Gradient Descent.

**A:** Gradient Descent is an optimization algorithm that iteratively updates model parameters to minimize the cost function.

```
θ = θ − α * ∂J/∂θ
```
- `α` = learning rate (step size)
- `∂J/∂θ` = gradient of cost with respect to parameter

Types:
- **Batch GD** — uses all data per update (slow, stable)
- **Stochastic GD** — uses 1 sample per update (fast, noisy)
- **Mini-batch GD** — uses a small batch (best of both)

---

### Q5. What is R-squared and Adjusted R-squared?

**A:**
- **R²** measures the proportion of variance in the target explained by the model:
```
R² = 1 - (SS_res / SS_tot)
```
Range: 0 to 1. Higher is better. R² = 1 means perfect fit.

- **Adjusted R²** penalizes for adding irrelevant features:
```
Adj R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]
```
Where `k` = number of features. Use Adjusted R² when comparing models with different numbers of features.

---

### Q6. What is multicollinearity?

**A:** Multicollinearity occurs when two or more independent variables are highly correlated with each other. This makes it difficult to determine the individual effect of each feature on the target.

**Problem:** Coefficients become unstable, have high standard errors, and may have incorrect signs.

---

### Q7. How do you detect multicollinearity?

**A:**
1. **Correlation matrix** — check for pairwise correlations > 0.8
2. **VIF (Variance Inflation Factor)**:
   - VIF = 1: No multicollinearity
   - VIF 1–5: Moderate
   - VIF > 10: Severe multicollinearity
3. **Condition number** of the feature matrix

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

---

### Q8. What is heteroscedasticity?

**A:** Heteroscedasticity means the variance of residuals is not constant across all levels of the independent variables. For example, residuals spread wider as the predicted value increases.

**Detection:** Residual vs fitted plot — if the spread fans out, heteroscedasticity exists.

**Fix:** Log transform the target, use weighted least squares, or use robust standard errors.

---

### Q9. What is the difference between correlation and regression?

**A:**
| | Correlation | Regression |
|---|---|---|
| Purpose | Measures strength of relationship | Predicts one variable from another |
| Output | Single number (−1 to 1) | Equation with coefficients |
| Direction | Symmetric (X↔Y same as Y↔X) | Asymmetric (X predicts Y) |
| Use case | Exploration | Prediction/inference |

---

### Q10. How do outliers affect Linear Regression?

**A:** Outliers have a large influence on the regression line because MSE squares the error, so large errors get disproportionately large weight. A single outlier can shift the entire regression line.

**Detection:** Use Cook's Distance, studentized residuals, or leverage scores.
**Fix:** Remove verified data entry errors, use Huber loss (robust regression), or apply log transformation.

---

### Q11. What is the Normal Equation?

**A:** A closed-form solution to find optimal weights without gradient descent:
```
β = (XᵀX)⁻¹ Xᵀy
```
**Pros:** Exact solution, no learning rate needed.
**Cons:** O(n³) matrix inversion — impractical for large datasets (n > 10,000 features).

---

### Q12. What is Ridge Regression?

**A:** Ridge (L2 regularization) adds the sum of squared coefficients to the cost function:
```
Cost = MSE + λ Σβᵢ²
```
- Shrinks coefficients toward zero but never exactly zero
- Works well when many features have small effects
- `λ` controls regularization strength (higher λ = more shrinkage)

---

### Q13. What is Lasso Regression?

**A:** Lasso (L1 regularization) adds the sum of absolute coefficients:
```
Cost = MSE + λ Σ|βᵢ|
```
- Can shrink coefficients **exactly to zero** → performs feature selection
- Preferred when only a few features are truly important

---

### Q14. What is the difference between Ridge and Lasso?

**A:**
| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | Sum of squares | Sum of absolutes |
| Coefficients | Shrinks toward 0 | Can be exactly 0 |
| Feature selection | No | Yes |
| Best for | Many small effects | Few large effects |
| Solution | Closed-form exists | No closed-form |

---

### Q15. What is Elastic Net?

**A:** Elastic Net combines both L1 and L2 penalties:
```
Cost = MSE + λ₁ Σ|βᵢ| + λ₂ Σβᵢ²
```
Best of both worlds — does feature selection (like Lasso) while handling correlated features (like Ridge). Controlled by `l1_ratio` in sklearn.

---

### Q16. What is feature scaling and why is it needed?

**A:** Feature scaling puts all features on a similar scale. Linear Regression with gradient descent converges much faster when features are scaled because the gradient steps are more balanced.

- **Standardization (Z-score):** `x = (x − mean) / std` → mean=0, std=1
- **Normalization (Min-Max):** `x = (x − min) / (max − min)` → range [0,1]

Note: The Normal Equation doesn't require scaling.

---

### Q17. What is a residual?

**A:** A residual is the difference between the actual and predicted value:
```
Residual = yᵢ − ŷᵢ
```
Positive residual = model underestimated. Negative = overestimated. The sum of all residuals in OLS is always 0.

---

### Q18. What is polynomial regression?

**A:** Polynomial regression extends linear regression by adding polynomial terms of the features (e.g., x², x³). It fits non-linear relationships while still being a linear model in terms of its coefficients.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```
**Risk:** High degree polynomials overfit easily.

---

### Q19. What is the difference between simple and multiple regression?

**A:**
- **Simple Linear Regression:** One feature → `ŷ = β₀ + β₁x`
- **Multiple Linear Regression:** Multiple features → `ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

---

### Q20. What is underfitting?

**A:** Underfitting occurs when the model is too simple to capture the underlying pattern in the data. It has high bias and low variance. Signs: high training error and high test error.

**Fix:** Add more features, use polynomial features, reduce regularization, or use a more complex model.

---

## Quick Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Small dataset
X = np.array([[1400,3,10],[1600,3,5],[1700,4,8],[1875,4,3],[1100,2,15]])
y = np.array([245, 312, 279, 308, 199])

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R²:", r2_score(y, model.predict(X)))
```

---

## Additional Questions & Answers (Q21–Q50)

---

### Q21. What is the difference between MSE and MAE as a loss function?

**A:**
| | MSE | MAE |
|---|---|---|
| Formula | `(1/n) Σ(y−ŷ)²` | `(1/n) Σ\|y−ŷ\|` |
| Outlier sensitivity | High (squares errors) | Low (linear errors) |
| Differentiable | Always | Not at 0 |
| Optimal prediction | Mean | Median |
| Use when | Outliers are rare and important | Outliers are common/ignored |

MSE is more common because it is differentiable everywhere, making gradient descent cleaner.

---

### Q22. What is the Gauss-Markov theorem?

**A:** The Gauss-Markov theorem states that under the OLS assumptions (linearity, homoscedasticity, independence, no multicollinearity), the OLS estimator is **BLUE** — Best Linear Unbiased Estimator:
- **Best:** Lowest variance among all linear unbiased estimators
- **Linear:** Estimator is a linear function of the response
- **Unbiased:** E[β̂] = β (correct on average)

This is why OLS is the go-to method when assumptions hold.

---

### Q23. What is the difference between parametric and non-parametric models?

**A:**
| | Parametric | Non-Parametric |
|---|---|---|
| Assumptions | Strong (e.g., linearity) | Few or none |
| Parameters | Fixed number | Grows with data |
| Examples | Linear Regression | KNN, Decision Tree |
| Data need | Less data needed | More data needed |
| Interpretability | High | Lower |

Linear Regression is parametric — it assumes a linear relationship and has a fixed set of β coefficients.

---

### Q24. Can Linear Regression be used for classification?

**A:** Technically yes, but it is a poor choice:
- Predictions can fall outside [0,1] — not valid probabilities
- Decision boundary is unstable for unbalanced or distant classes
- Loss function (MSE) is not appropriate for 0/1 labels

The only acceptable use: **LDA (Linear Discriminant Analysis)** under very specific conditions. Otherwise, use Logistic Regression for classification.

---

### Q25. What happens when you have more features than observations (p > n)?

**A:** The Normal Equation (XᵀX)⁻¹ becomes singular — non-invertible. OLS has no unique solution; the system is underdetermined (infinitely many solutions).

**Fixes:**
1. **Ridge Regression:** (XᵀX + λI)⁻¹ is always invertible — adds λ to diagonal
2. **Lasso / Elastic Net:** Performs feature selection to reduce to fewer features
3. **Dimensionality reduction:** PCA first, then regress on components
4. **Regularized solvers:** `solver='saga'` in sklearn handles p > n

---

### Q26. What is weighted least squares?

**A:** Weighted Least Squares (WLS) assigns different weights to observations — useful when variance is not constant (heteroscedasticity):

```
Minimize: Σ wᵢ (yᵢ − ŷᵢ)²
```

Observations with **lower variance** get **higher weight** (trusted more). WLS is the efficient estimator under heteroscedasticity.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y, sample_weight=weights)
```

---

### Q27. What is the hat matrix (projection matrix)?

**A:** The hat matrix H projects y onto the column space of X:
```
ŷ = Hy = X(XᵀX)⁻¹Xᵀ y
```

Diagonal elements hᵢᵢ are **leverage scores** — how much influence observation i has on its own predicted value:
- hᵢᵢ = 1: Observation perfectly determines its own prediction
- hᵢᵢ > 2(p+1)/n: High leverage point (potential influential outlier)

---

### Q28. What is Cook's Distance?

**A:** Cook's Distance measures how much the fitted values change if observation i is removed:

```
Dᵢ = Σⱼ (ŷⱼ − ŷⱼ₍ᵢ₎)² / (p × MSE)
```

- Dᵢ > 1: Highly influential point — investigate further
- Dᵢ > 4/n: Common threshold for potential influence

```python
import statsmodels.api as sm
model = sm.OLS(y, sm.add_constant(X)).fit()
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
```

---

### Q29. What is the difference between confidence interval and prediction interval?

**A:**
| | Confidence Interval | Prediction Interval |
|---|---|---|
| Estimates | Mean response at x | Individual new observation at x |
| Width | Narrower | Wider (includes individual noise) |
| Formula | `ŷ ± t × SE(ŷ)` | `ŷ ± t × √(SE(ŷ)² + σ²)` |
| Use | "Average house price at 1500 sqft" | "Price of this specific house at 1500 sqft" |

Prediction intervals are always wider because they account for both uncertainty in the mean AND individual variability.

---

### Q30. What is stepwise regression?

**A:** Stepwise regression is an automated feature selection method:

- **Forward selection:** Start empty, add one feature at a time (best AIC/BIC improvement)
- **Backward elimination:** Start full model, remove features one at a time
- **Bidirectional:** Combination of forward and backward

**Criticism:**
- Multiple testing problem — inflates false discovery rate
- Final model depends on order of addition/removal
- Better alternatives: Lasso, cross-validated feature selection

---

### Q31. What is the variance inflation factor (VIF) formula?

**A:**
```
VIF(j) = 1 / (1 − R²ⱼ)
```

Where R²ⱼ is the R² from regressing feature j on all other features.

Interpretation:
- VIF = 1: No multicollinearity
- VIF = 5: R²ⱼ = 0.8 → feature j is 80% explained by others
- VIF = 10: R²ⱼ = 0.9 → severe multicollinearity

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

---

### Q32. What is the bias-variance decomposition for Linear Regression?

**A:**
```
E[(y − ŷ)²] = Bias² + Variance + Irreducible Noise
```

For OLS:
- **Bias = 0** (OLS is unbiased under Gauss-Markov assumptions)
- **Variance** = σ²(XᵀX)⁻¹ — decreases with more data, increases with multicollinearity

For Ridge:
- **Bias > 0** (introduces bias by shrinking coefficients)
- **Variance < OLS** (smaller — that's the point of Ridge)

The regularization trade-off: accept some bias to reduce variance.

---

### Q33. What is the intercept term and when should you suppress it?

**A:** The intercept β₀ represents the predicted value when all features are 0. It should **almost always be included** — suppressing it forces the regression line through the origin.

**Suppress intercept only when:**
- Domain knowledge confirms prediction = 0 when all inputs = 0
- Working with differenced time series or some specific physics models

```python
# With intercept (default)
LinearRegression(fit_intercept=True)
# Without intercept (rare)
LinearRegression(fit_intercept=False)
```

---

### Q34. How do you test for homoscedasticity?

**A:**
1. **Residual vs fitted plot:** Residuals should be evenly scattered — no funnel shape
2. **Breusch-Pagan test:** Tests if residual variance is function of features
3. **White test:** More general test for heteroscedasticity

```python
import statsmodels.stats.diagnostic as diag
model = sm.OLS(y, sm.add_constant(X)).fit()
bp_test = diag.het_breuschpagan(model.resid, model.model.exog)
print("Breusch-Pagan p-value:", bp_test[1])
# p < 0.05 → heteroscedasticity detected
```

---

### Q35. What is the difference between in-sample and out-of-sample evaluation?

**A:**
- **In-sample (training):** Evaluate model on data used to train it — optimistic, not trustworthy
- **Out-of-sample (test/validation):** Evaluate on unseen data — realistic generalization estimate

Overfitting makes in-sample R² near 1.0 while out-of-sample R² is much lower. Always report out-of-sample metrics.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("CV R²:", scores.mean(), "±", scores.std())
```

---

### Q36. What is the impact of irrelevant features on Linear Regression?

**A:**
- Irrelevant features add noise to the model but do not bias OLS estimates
- They increase variance (overfitting) by using up degrees of freedom
- Adjusted R² penalizes for irrelevant features (R² never decreases with more features, but Adjusted R² can)
- Lasso automatically zeros out irrelevant feature coefficients

---

### Q37. What is multivariate regression vs multiple regression?

**A:**
- **Multiple regression:** One target variable, multiple features → `y = Xβ`
- **Multivariate regression:** Multiple target variables simultaneously → `Y = XB` where Y is a matrix

Multivariate regression fits multiple response variables jointly and can model correlations between responses.

```python
# Multivariate regression (sklearn)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, Y)  # Y has multiple columns
```

---

### Q38. How would you diagnose a poorly fitting Linear Regression model?

**A:** Systematic diagnosis:
1. **Residual vs fitted plot:** Non-random pattern → non-linearity or heteroscedasticity
2. **QQ plot of residuals:** Deviation from line → non-normality
3. **Scale-location plot:** Heteroscedasticity (funnel shape)
4. **Residuals vs leverage:** Influential outliers (Cook's distance)
5. **Low R²:** Model doesn't explain variance — missing features or wrong functional form
6. **High RMSE relative to target range:** Poor prediction accuracy

---

### Q39. What is the difference between OLS and GLS?

**A:**
| | OLS (Ordinary Least Squares) | GLS (Generalized Least Squares) |
|---|---|---|
| Residual covariance | Assumes diagonal (independent, equal variance) | Handles arbitrary covariance structure |
| Efficient when | Homoscedastic, independent errors | Correlated or heteroscedastic errors |
| Special cases | — | WLS (weighted), FGLS (feasible GLS) |

GLS: `β_GLS = (XᵀΩ⁻¹X)⁻¹ XᵀΩ⁻¹y` where Ω is the error covariance matrix.

---

### Q40. What is quantile regression?

**A:** Quantile regression estimates the conditional **quantile** (e.g., median, 90th percentile) of the target rather than the mean:

```
Minimize: Σ ρτ(yᵢ − ŷᵢ)
where ρτ(u) = u(τ − I(u<0))  — asymmetric loss
```

**Uses:**
- Estimate median (robust to outliers) → τ = 0.5
- Estimate upper/lower bounds → τ = 0.9 or 0.1
- Prediction intervals without distributional assumptions
- Risk modeling (Value at Risk in finance)

```python
from sklearn.linear_model import QuantileRegressor
model = QuantileRegressor(quantile=0.9)
```

---

### Q41. What is cross-validation and why is it important for regression?

**A:** Cross-validation estimates generalization performance by rotating the test set:

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf,
                          scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-scores.mean())
print("CV RMSE:", rmse_cv)
```

For regression: use **5-fold or 10-fold CV**. Report mean ± std of RMSE/R² across folds. Prevents over-optimistic evaluation from a single train-test split.

---

### Q42. What is the difference between standardization and mean normalization?

**A:**
- **Standardization (Z-score):** `x = (x − μ) / σ` → mean=0, std=1
- **Mean normalization:** `x = (x − μ) / (max − min)` → mean=0, range [−1, 1]
- **Min-Max normalization:** `x = (x − min) / (max − min)` → range [0, 1]

For Linear Regression with gradient descent: standardization is preferred — it handles outliers better and makes learning rate selection easier.

---

### Q43. What is leverage vs influence in regression?

**A:**
- **Leverage:** How far a data point is from the center of X space — high leverage = unusual feature values. Measured by hᵢᵢ (diagonal of hat matrix).
- **Influence:** How much the regression coefficients change if the point is removed. Measured by Cook's Distance.

A point can have:
- High leverage + fits regression line → not influential
- High leverage + far from regression line → very influential
- Low leverage + large residual → outlier but not influential

---

### Q44. What is regularization path?

**A:** The regularization path shows how coefficients change as the regularization strength λ varies from 0 to ∞:

```python
from sklearn.linear_model import LassoCV, lasso_path
import numpy as np

alphas, coefs, _ = lasso_path(X, y, alphas=np.logspace(-3, 1, 50))
# Plot: each line = one feature coefficient, x-axis = lambda
```

- At λ=0: Full OLS solution
- As λ increases: Coefficients shrink (Ridge) or drop to zero (Lasso)
- Used to understand feature importance and select optimal λ

---

### Q45. What is the condition number and why does it matter?

**A:** The condition number κ(X) = σ_max/σ_min (ratio of largest to smallest singular value of X):

- κ ≈ 1: Well-conditioned, stable computation
- κ > 1000: Ill-conditioned — small data changes cause large coefficient changes
- κ > 10⁶: Severe — numerical instability, essentially singular

High condition number → multicollinearity → unstable coefficient estimates. Ridge regression improves condition number: κ(XᵀX + λI).

---

### Q46. How does Linear Regression extend to time series (autoregression)?

**A:** In an **AR(p) model**, the target is regressed on its own past values:

```
yₜ = β₀ + β₁yₜ₋₁ + β₂yₜ₋₂ + ... + βₚyₜ₋ₚ + εₜ
```

This is just Linear Regression where the features are lagged values of the target. Key difference: observations are NOT independent — violates OLS assumption. Use time-series cross-validation (no data leakage from future to past).

---

### Q47. What is robust regression and when do you use it?

**A:** Robust regression is less sensitive to outliers than OLS by using a loss function that down-weights large residuals:

- **Huber regression:** MSE for small residuals, MAE for large ones
- **RANSAC:** Randomly sample subsets, fit on inlier-only subset
- **Theil-Sen:** Uses median of pairwise slopes

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor

huber = HuberRegressor(epsilon=1.35)  # epsilon controls transition MSE→MAE
ransac = RANSACRegressor(min_samples=0.5, residual_threshold=10)
```

Use when: data has known outliers that cannot be removed.

---

### Q48. What does it mean for a regression model to be interpretable?

**A:** A Linear Regression model is interpretable because:
1. Each coefficient β has a direct meaning: "1-unit increase in x → β change in y, all else equal"
2. The sign tells you direction (positive/negative effect)
3. Coefficients can be compared after standardizing features
4. p-values indicate which features are statistically significant

Contrast with neural networks: predictions are hard to attribute to specific inputs.

---

### Q49. How do you handle categorical variables in Linear Regression?

**A:** Linear Regression requires numerical inputs — categorical variables must be encoded:

```python
# One-hot encoding (preferred for nominal categories)
pd.get_dummies(df['city'], drop_first=True)  # drop_first avoids dummy trap

# Label encoding (only for ordinal categories like Low/Medium/High)
from sklearn.preprocessing import OrdinalEncoder
```

**Dummy variable trap:** If you one-hot encode without dropping one category, columns are perfectly collinear → multicollinearity. Always drop one category (reference category) or use `drop_first=True`.

---

### Q50. How do you interpret negative coefficients in Linear Regression?

**A:** A negative coefficient means the feature has an **inverse relationship** with the target — as the feature increases, the predicted target decreases (holding all other features constant).

Example from our house dataset:
- **Age coefficient = −5.2:** Older houses sell for $5,200 less per year of age, holding size and bedrooms constant
- This makes intuitive sense: newer houses are more desirable

**Important:** Interpret coefficients conditionally — "holding all other variables constant." Marginal effect depends on feature correlations.

---

## Extended Code Examples

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dataset
X = np.array([[1400,3,10],[1600,3,5],[1700,4,8],[1875,4,3],
              [1100,2,15],[1550,3,7],[2350,4,2],[2450,4,1]])
y = np.array([245, 312, 279, 308, 199, 219, 405, 324])

# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Compare OLS, Ridge, Lasso ---
models = {
    'OLS':   LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.5),
    'Huber': HuberRegressor(epsilon=1.35)
}
for name, model in models.items():
    cv = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')
    model.fit(X_scaled, y)
    print(f"{name}: CV R²={cv.mean():.3f}±{cv.std():.3f}  Coefs={model.coef_.round(2)}")

# --- Residual analysis ---
lr = LinearRegression()
lr.fit(X_scaled, y)
residuals = y - lr.predict(X_scaled)
print("\nResidual mean (should be ~0):", residuals.mean().round(4))
print("Residual std:", residuals.std().round(2))

# --- Regularization path for Lasso ---
from sklearn.linear_model import lasso_path
alphas, coefs, _ = lasso_path(X_scaled, y, alphas=np.logspace(-2, 2, 50))
print("\nLasso path shapes (alphas, coefs):", alphas.shape, coefs.shape)

# --- Polynomial features ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
lr_poly = LinearRegression()
cv_poly = cross_val_score(lr_poly, X_poly, y, cv=3, scoring='r2')
print(f"\nPolynomial (degree=2) CV R²: {cv_poly.mean():.3f}")

# --- VIF check ---
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
features = ['Size', 'Bedrooms', 'Age']
for f, v in zip(features, vif):
    print(f"VIF {f}: {v:.2f}")
```
