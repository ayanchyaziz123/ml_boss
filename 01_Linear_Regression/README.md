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
