# PCA & Dimensionality Reduction — Interview Questions & Answers

---

## Small Dataset Example

| Height (cm) | Weight (kg) | BMI | Waist (cm) | Hip (cm) |
|-------------|-------------|-----|------------|----------|
| 170 | 65 | 22.5 | 80 | 95 |
| 160 | 55 | 21.5 | 72 | 88 |
| 180 | 85 | 26.2 | 92 | 102 |
| 175 | 75 | 24.5 | 86 | 98 |
| 165 | 60 | 22.0 | 75 | 90 |
| 185 | 95 | 27.8 | 98 | 108 |
| 155 | 50 | 20.8 | 68 | 85 |
| 172 | 70 | 23.7 | 83 | 97 |

### Dataset Questions

**Q: Features Height, Weight, BMI, Waist, Hip are all correlated. What does PCA do with these 5 features?**
> **A:** PCA finds new axes (principal components) that capture the maximum variance. PC1 likely captures the general "body size" direction (all features positive, since tall people tend to weigh more with larger waist/hip). PC2 might capture body shape ratio (waist-hip ratio independent of size). The first 1–2 PCs likely explain >90% of variance, allowing compression from 5D to 2D.

**Q: BMI = Weight / Height². If you include all 5 features, is BMI redundant?**
> **A:** Yes — BMI is a deterministic function of Height and Weight, so it adds no new information. PCA would detect this: BMI's variance is fully explained by the Height-Weight component. In practice, you'd remove BMI before PCA (or let PCA handle it) to avoid multicollinearity distorting the explained variance ratios.

**Q: You apply PCA and the first PC explains 85% variance, the second 10%. Should you use 1 or 2 components?**
> **A:** If 85% is sufficient for your downstream task, use 1 component. If you want to retain ≥95% of information, use 2 components (85+10=95%). The decision depends on how much information loss your model can tolerate.

---

## Questions & Answers

---

### Q1. What is PCA?

**A:** Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique that transforms features into a new orthogonal coordinate system where:
- The first axis (PC1) captures the most variance
- The second axis (PC2) captures the next most variance, orthogonal to PC1
- And so on...

PCA compresses n features into k < n components while retaining maximum information (variance).

---

### Q2. What is an eigenvector?

**A:** An eigenvector of a matrix A is a non-zero vector v that satisfies:
```
Av = λv
```
- v doesn't change direction when multiplied by A — only scales
- In PCA: eigenvectors of the covariance matrix = **principal component directions**
- Each eigenvector points in the direction of maximum remaining variance

---

### Q3. What is an eigenvalue?

**A:** An eigenvalue λ is the scalar in `Av = λv`:
- In PCA: eigenvalue = amount of variance explained by its eigenvector (principal component)
- Larger eigenvalue → that PC captures more variance
- Eigenvalues are sorted in descending order in PCA

```
Explained variance ratio of PCᵢ = λᵢ / Σλⱼ
```

---

### Q4. What is explained variance?

**A:** Explained variance tells how much of the total variance in the data is captured by each principal component:

```python
pca = PCA(n_components=3)
pca.fit(X_scaled)
print(pca.explained_variance_ratio_)         # e.g., [0.72, 0.18, 0.06]
print(pca.explained_variance_ratio_.cumsum()) # cumulative: [0.72, 0.90, 0.96]
```

**How to choose n_components:** Pick the smallest k where cumulative explained variance ≥ 95% (or your threshold).

---

### Q5. What is the difference between PCA and LDA?

**A:**
| | PCA | LDA |
|---|---|---|
| Type | Unsupervised | Supervised |
| Goal | Maximize variance | Maximize class separability |
| Uses labels | No | Yes |
| Components | min(n_features, n_samples) − 1 | n_classes − 1 |
| Best for | Compression/visualization | Classification preprocessing |
| Handles multimodal | No | Better |

LDA is better for classification preprocessing; PCA is better for general compression.

---

### Q6. What is t-SNE?

**A:** t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique for **visualization** (typically 2D or 3D):
- Preserves local structure: similar points in high-D → nearby in 2D
- Uses probability distributions over pairwise distances + t-distribution
- Non-deterministic, no projection function

**Key parameters:**
- `perplexity` (5–50): controls neighborhood size
- `learning_rate` (10–1000): step size

**Limitations:** Very slow O(n²), not for compression (no projection), different runs give different results.

---

### Q7. What is UMAP?

**A:** UMAP (Uniform Manifold Approximation and Projection) is a newer dimensionality reduction technique:
- Preserves both local AND global structure (better than t-SNE)
- Much faster than t-SNE (O(n log n))
- Has a transform function → can project new data
- Better for downstream ML tasks

```python
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
X_2d = reducer.fit_transform(X_scaled)
```

---

### Q8. What is feature selection vs feature extraction?

**A:**
| | Feature Selection | Feature Extraction |
|---|---|---|
| Method | Select subset of original features | Create new features from originals |
| Interpretability | High (original features kept) | Lower (new abstract features) |
| Example algorithms | LASSO, RFE, mutual information | PCA, autoencoders, t-SNE |
| Keeps original features | Yes | No |
| Best for | Interpretable models | Dense high-dimensional data |

---

### Q9. What is the curse of dimensionality (in PCA context)?

**A:** In high dimensions:
- Data becomes sparse — points are far apart
- Distance metrics lose meaning
- ML models overfit easily
- Visualization impossible

PCA helps by reducing dimensions while keeping maximum information. If 95% of variance is captured in 2 PCs, working in 2D is much more tractable than 50D.

---

### Q10. What is Kernel PCA?

**A:** Kernel PCA extends PCA to non-linear relationships using the kernel trick:
- Maps data to high-dimensional space via kernel function
- Applies PCA in that space
- Returns non-linear projections in original space

```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X)
```

Useful when data has non-linear structure that linear PCA cannot capture.

---

### Q11. When should you use PCA?

**A:** Use PCA when:
1. Features are **highly correlated** (multicollinearity)
2. Data has **high dimensionality** and you want to speed up training
3. You want to **visualize** high-dimensional data (reduce to 2–3 components)
4. You want to reduce **overfitting** by reducing feature count
5. **Preprocessing for models** sensitive to correlated features

Do NOT use PCA when:
- Interpretability of original features is required
- Features are already independent
- Non-linear structure is important (use Kernel PCA or autoencoders)

---

### Q12. Does PCA require feature scaling?

**A:** YES — always scale before PCA. PCA maximizes variance; features with larger scales will dominate:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

Without scaling, a feature in $100,000s would completely dominate a feature in 0–1 range.

---

### Q13. What is the relationship between PCA and SVD?

**A:** PCA and SVD are mathematically equivalent:
```
X = U Σ Vᵀ  (SVD of data matrix X)
```
- **V** columns = principal component directions (eigenvectors of XᵀX)
- **Σ** diagonal = singular values (square root of eigenvalues)
- **U Σ** = principal component scores

Sklearn uses SVD internally (more numerically stable than direct eigendecomposition):
```python
PCA(svd_solver='full')  # or 'randomized' for large data
```

---

### Q14. What is Truncated SVD?

**A:** Truncated SVD computes only the top k singular vectors without computing the full decomposition. Much faster for large, sparse matrices:

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
X_reduced = svd.fit_transform(sparse_matrix)
```

Used in text analysis (LSA — Latent Semantic Analysis). Unlike PCA, doesn't require centering the data (works on sparse matrices).

---

### Q15. What is the difference between PCA and autoencoders?

**A:**
| | PCA | Autoencoder |
|---|---|---|
| Type | Linear transformation | Non-linear (with activation functions) |
| Optimization | Closed-form (SVD) | Gradient descent |
| Components | Orthogonal | May not be |
| Captures | Linear structure | Non-linear structure |
| Speed | Fast | Slow (neural network training) |
| Interpretability | Good (explained variance) | Lower |

Linear autoencoder (no activation) learns the same subspace as PCA.

---

### Q16. What is ICA (Independent Component Analysis)?

**A:** ICA separates mixed signals into statistically independent sources. Unlike PCA (maximizes variance), ICA maximizes statistical independence:

```
X = A × S  → find A and S such that S components are independent
```

Use case: Blind source separation (separating mixed audio signals), EEG signal decomposition.

PCA components are uncorrelated but not necessarily independent. ICA components are maximally independent.

---

### Q17. What is Incremental PCA?

**A:** Incremental PCA processes data in mini-batches — useful when data doesn't fit in memory:

```python
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=10, batch_size=100)
for batch in data_batches:
    ipca.partial_fit(batch)
X_reduced = ipca.transform(X)
```

Same result as standard PCA but memory-efficient for large datasets.

---

### Q18. How do you choose the number of principal components?

**A:** Multiple methods:
1. **Cumulative explained variance:** Pick k where Σ explained variance ≥ 95%
2. **Scree plot:** Plot eigenvalues, look for "elbow" where they flatten
3. **Kaiser criterion:** Keep components with eigenvalue > 1
4. **Cross-validation:** Test downstream task performance for different k values

```python
pca = PCA().fit(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumvar >= 0.95) + 1
print(f"Components for 95% variance: {k}")
```

---

### Q19. What is the difference between PCA and t-SNE for visualization?

**A:**
| | PCA | t-SNE |
|---|---|---|
| Type | Linear | Non-linear |
| Speed | Very fast | Slow O(n²) |
| Global structure | Preserved | Sometimes distorted |
| Local structure | May overlap | Well-preserved |
| New data projection | Yes | No |
| Deterministic | Yes | No (random seed) |
| Best for | Quick overview | Cluster visualization |

For exploratory visualization: use t-SNE or UMAP. For preprocessing: use PCA.

---

### Q20. What are the limitations of PCA?

**A:**
1. **Linear only** — can't capture non-linear relationships
2. **Loses interpretability** — PCs are linear combinations of all original features
3. **Assumes Gaussian distribution** — poorly handles multimodal data
4. **Sensitive to outliers** — outliers inflate variance, distort components
5. **Requires scaling** — otherwise high-variance features dominate
6. **Covariance only** — ignores class labels (unlike LDA)
7. **Components aren't necessarily meaningful** — hard to interpret what PC1 means

---

## Quick Code Example

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X = np.array([[170,65,22.5,80,95],[160,55,21.5,72,88],
              [180,85,26.2,92,102],[175,75,24.5,86,98],
              [165,60,22.0,75,90],[185,95,27.8,98,108],
              [155,50,20.8,68,85],[172,70,23.7,83,97]])

# Step 1: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Fit PCA
pca = PCA()
pca.fit(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_).round(3))

# Choose n_components for 95% variance
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components}")

# Step 3: Transform
pca_final = PCA(n_components=n_components)
X_reduced = pca_final.fit_transform(X_scaled)
print(f"Original shape: {X_scaled.shape} → Reduced: {X_reduced.shape}")

# Component loadings (which features contribute to each PC)
feature_names = ['Height', 'Weight', 'BMI', 'Waist', 'Hip']
for i, comp in enumerate(pca_final.components_):
    print(f"\nPC{i+1} loadings:")
    for feat, loading in zip(feature_names, comp):
        print(f"  {feat}: {loading:.3f}")
```
