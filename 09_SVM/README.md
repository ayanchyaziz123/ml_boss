# Support Vector Machine (SVM) — Interview Questions & Answers

---

## Small Dataset Example

| Feature 1 (x₁) | Feature 2 (x₂) | Class |
|-----------------|-----------------|-------|
| 1.0 | 2.0 | -1 |
| 1.5 | 1.8 | -1 |
| 2.0 | 3.0 | -1 |
| 5.0 | 5.0 | +1 |
| 5.5 | 4.5 | +1 |
| 6.0 | 5.0 | +1 |
| 3.0 | 3.5 | -1 |
| 4.0 | 4.5 | +1 |

### Dataset Questions

**Q: In this 2D dataset, what does the SVM try to find?**
> **A:** SVM finds the **maximum margin hyperplane** — a line (in 2D) that separates the two classes (-1 and +1) with the **largest possible margin** (distance between the hyperplane and the nearest points of each class). The nearest points are called **support vectors**.

**Q: If you add C=0.01 (very small), what happens to the decision boundary?**
> **A:** Small C allows more margin violations — the SVM prioritizes a wider margin even if some points are misclassified. This creates a **smoother, more generalized** boundary (underfits). Large C → narrow margin, fewer violations → may overfit.

**Q: Points (1.0, 2.0), (5.0, 5.0), and (4.0, 4.5) are the support vectors. If you remove a non-support-vector point, does the decision boundary change?**
> **A:** No. The SVM decision boundary depends **only on support vectors**. Removing any other point has zero effect on the trained model. This is why SVM is memory efficient for sparse data.

---

## Questions & Answers

---

### Q1. What is SVM?

**A:** Support Vector Machine (SVM) is a supervised learning algorithm that finds the **optimal hyperplane** to separate classes with the **maximum margin**. 

Key idea: Among all hyperplanes that separate the classes, find the one with the largest margin — this maximizes generalization. The decision function only depends on a subset of training points called **support vectors**.

---

### Q2. What is a hyperplane?

**A:** A hyperplane is an n-1 dimensional flat affine subspace that divides the feature space:
- In 2D: a line (`w₁x₁ + w₂x₂ + b = 0`)
- In 3D: a plane
- In nD: a hyperplane (`w·x + b = 0`)

SVM finds the hyperplane defined by weight vector **w** and bias **b** that maximizes the separation between classes.

---

### Q3. What is the margin in SVM?

**A:** The margin is the distance between the decision hyperplane and the closest training points from each class (support vectors).

```
Margin = 2 / ||w||
```

Maximizing the margin = minimizing `||w||²`. A larger margin generally means better generalization because the model is less sensitive to individual training points.

---

### Q4. What are support vectors?

**A:** Support vectors are the training samples closest to the decision boundary — they "support" (define) the hyperplane. Key properties:
- Only support vectors determine the decision boundary
- Removing non-support-vector points doesn't change the model
- Typically a small subset of training data (sparse representation)
- If training data changes but support vectors remain the same, the model is unchanged

---

### Q5. What is the kernel trick?

**A:** The kernel trick allows SVM to create non-linear decision boundaries without explicitly computing the high-dimensional feature space. 

Instead of computing `φ(x)·φ(z)` (expensive), compute `K(x, z) = φ(x)·φ(z)` directly:
```
Linear:      K(x,z) = x·z
RBF:         K(x,z) = exp(−γ||x−z||²)
Polynomial:  K(x,z) = (γx·z + r)^d
Sigmoid:     K(x,z) = tanh(γx·z + r)
```

The SVM only needs dot products → swap them with kernel function → works in infinite-dimensional spaces implicitly.

---

### Q6. What is the RBF (Radial Basis Function) kernel?

**A:**
```
K(x, z) = exp(−γ ||x − z||²)
```
- Measures similarity by distance: similar points → K near 1, dissimilar → K near 0
- Creates circular/elliptical decision boundaries
- Most widely used kernel for non-linear SVM
- `γ` controls the "influence radius" of each training point

**γ too large:** Each point only influences its local neighborhood → complex boundary → overfitting
**γ too small:** Points influence far regions → smooth boundary → underfitting

---

### Q7. What is the linear kernel?

**A:**
```
K(x, z) = x · z
```
Equivalent to standard SVM with no kernel. Use when:
- Data is linearly separable
- Very high-dimensional data (e.g., text with 10,000+ features)
- Speed is important (much faster than RBF)
- `n_features >> n_samples`

---

### Q8. What is the polynomial kernel?

**A:**
```
K(x, z) = (γ x·z + r)^d
```
Creates polynomial decision boundaries of degree d. `d=2` gives quadratic boundaries. More interpretable than RBF but less flexible. Parameters: `degree`, `gamma`, `coef0` (r).

---

### Q9. What is the C parameter in SVM?

**A:** C is the regularization parameter (penalty for misclassification):

```
Minimize: (1/2)||w||² + C Σ ξᵢ
```
- **Small C:** Large margin allowed (even with misclassifications) → simpler model → may underfit
- **Large C:** Fewer misclassifications tolerated → smaller margin → may overfit

C controls the **bias-variance trade-off**: larger C = lower bias, higher variance.

---

### Q10. What is gamma in SVM?

**A:** `gamma` controls the influence of each training sample in the RBF kernel:

- **Large gamma:** Each point has very local influence → complex, wiggly boundary → overfitting
- **Small gamma:** Each point influences distant regions → smooth boundary → underfitting

For RBF kernel: `gamma = 'scale'` (default) = `1/(n_features × X.var())` is a good starting point.

---

### Q11. What is a hard margin SVM?

**A:** Hard margin SVM assumes data is **linearly separable** and requires zero misclassifications. It finds the exact maximum margin hyperplane with no slack.

**Problem:** If data is not perfectly separable (or has outliers), hard margin SVM fails to find a solution. Not practical for real-world data.

---

### Q12. What is a soft margin SVM?

**A:** Soft margin SVM (C-SVM) allows some misclassifications by introducing **slack variables** `ξᵢ ≥ 0`:

```
yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ
```
- `ξᵢ = 0`: Point is correctly classified with margin
- `0 < ξᵢ < 1`: Point is correctly classified but inside the margin
- `ξᵢ > 1`: Point is misclassified

C controls the trade-off between maximizing margin and minimizing total slack.

---

### Q13. Can SVM be used for regression?

**A:** Yes — **SVR (Support Vector Regression)** fits a tube (epsilon-insensitive zone) around the data:
- Points inside the tube: no penalty
- Points outside the tube: penalized proportional to distance from tube

```python
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)
```

---

### Q14. What is the difference between SVM and Logistic Regression?

**A:**
| | SVM | Logistic Regression |
|---|---|---|
| Loss function | Hinge loss | Cross-entropy loss |
| Output | Class label (no probability by default) | Probability (0–1) |
| Decision boundary | Maximizes margin | Passes through data center |
| Outlier sensitivity | Low (only SVs matter) | High |
| Non-linearity | Kernel trick | Feature engineering needed |
| Scales with data | Slow for large n | Faster |
| Interpretability | Less | More (coefficients = log-odds) |

---

### Q15. How does SVM handle non-linearly separable data?

**A:** Two approaches:
1. **Soft margin (C parameter):** Allow misclassifications for approximate linear separation
2. **Kernel trick:** Map data to higher dimensions where it IS linearly separable

```
Original space: circles and rings (non-separable)
RBF kernel maps to high-D space: linearly separable
```

---

### Q16. How do you choose the right kernel?

**A:**
| Data Type | Recommended Kernel |
|---|---|
| Linearly separable or high-dimensional sparse | Linear |
| Non-linear, medium-dimensional | RBF (start here) |
| Image/pixel data | RBF or Polynomial |
| Text data | Linear (TF-IDF features are high-dim) |
| Domain-specific similarity | Custom kernel |

**Rule:** Always try Linear first (fast). If performance is poor, try RBF with grid search on C and gamma.

---

### Q17. What are the advantages of SVM?

**A:**
1. Effective in high-dimensional spaces (even when d > n)
2. Memory efficient — only stores support vectors
3. Versatile — different kernels for different problems
4. Strong theoretical foundation (margin maximization)
5. Works well when there's a clear margin of separation
6. Robust to outliers (only support vectors matter)

---

### Q18. What are the disadvantages of SVM?

**A:**
1. **Slow on large datasets** — O(n²) to O(n³) training time
2. **No probability output** by default (use `probability=True` with Platt scaling)
3. **Sensitive to feature scaling** — must normalize/standardize features
4. **Hard to interpret** — especially with kernels
5. **Kernel choice** requires domain knowledge or expensive grid search
6. **Memory scales** with number of support vectors (can be large)

---

### Q19. How does SVM perform in high-dimensional spaces?

**A:** SVM performs well in high-dimensional spaces because:
- The margin maximization objective regularizes the model
- `n_features >> n_samples` is fine with linear kernel
- Common in text classification (vocab size 50,000+ features, few docs)

However, if `n_features >> n_samples` and `C` is too large, overfitting can still occur. Use L2 regularization (C parameter).

---

### Q20. How does SVM handle multi-class classification?

**A:** SVM is inherently binary. For multi-class, two strategies:

1. **One-vs-Rest (OvR):** Train K classifiers (each class vs all others), predict the class with highest confidence
2. **One-vs-One (OvO):** Train K(K-1)/2 classifiers (each pair of classes), majority vote

```python
from sklearn.svm import SVC
# decision_function_shape='ovr' (default) or 'ovo'
svm = SVC(kernel='rbf', decision_function_shape='ovr')
```

OvO is more accurate, OvR is faster to train.

---

## Quick Code Example

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

X = np.array([[1.0,2.0],[1.5,1.8],[2.0,3.0],[5.0,5.0],
              [5.5,4.5],[6.0,5.0],[3.0,3.5],[4.0,4.5]])
y = np.array([-1,-1,-1,1,1,1,-1,1])

# Feature scaling is important for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_scaled, y)
print("Support Vectors:", svm_linear.support_vectors_)
print("Number of SVs:", svm_linear.n_support_)

# RBF SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_rbf.fit(X_scaled, y)
print("\nRBF Accuracy:", svm_rbf.score(X_scaled, y))
print("Probabilities:", svm_rbf.predict_proba(X_scaled[:2]))
```
