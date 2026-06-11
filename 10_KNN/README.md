# K-Nearest Neighbors (KNN) — Interview Questions & Answers

---

## Small Dataset Example

| Height (cm) | Weight (kg) | Age | Fitness Level |
|-------------|-------------|-----|---------------|
| 170 | 65 | 25 | High |
| 165 | 80 | 30 | Low |
| 180 | 75 | 28 | High |
| 160 | 85 | 45 | Low |
| 175 | 70 | 32 | High |
| 155 | 90 | 50 | Low |
| 185 | 78 | 27 | High |
| 162 | 88 | 48 | Low |

### Dataset Questions

**Q: New person: Height=172, Weight=72, Age=29. Using k=3, what is the predicted fitness level?**
> **A:** Calculate Euclidean distances to all training points (after scaling). The 3 nearest neighbors would likely be: (170,65,25)=High, (175,70,32)=High, (180,75,28)=High. Majority vote → **High** fitness. (Note: scale features before computing distances!)

**Q: Why must you scale features before applying KNN to this dataset?**
> **A:** Height ranges 155–185 (30-unit range), Weight ranges 65–90 (25-unit range), but Age ranges 25–50 (25-unit range). Without scaling, all features contribute similarly. If we added income ($30,000–$100,000), it would dominate the distance calculation. Standardize with Z-score so each feature has equal influence.

**Q: If k=1, what is the training accuracy and what does that tell you?**
> **A:** k=1 means each training point's nearest neighbor is itself → 100% training accuracy. But this memorizes the data (maximum overfitting) and generalizes poorly. Use cross-validation to pick the real best k.

---

## Questions & Answers

---

### Q1. What is KNN?

**A:** K-Nearest Neighbors is a simple, non-parametric, instance-based learning algorithm that classifies a new point by looking at the k most similar training points and taking a majority vote (classification) or average (regression).

Key property: **no training phase** — the entire learning happens at prediction time. This is why it's called a "lazy learner."

---

### Q2. How does KNN work?

**A:** Step by step:
1. Store all training data
2. For a new point x:
   - Calculate distance from x to every training point
   - Find the k training points with smallest distance
   - **Classification:** Take majority class among k neighbors
   - **Regression:** Take mean value among k neighbors
3. Return prediction

Simple, no model parameters to learn. The "model" IS the training data.

---

### Q3. What is the value of K and how do you choose it?

**A:** K is the number of nearest neighbors to consider:
- **K=1:** Most flexible, memorizes training data (overfitting)
- **Large K:** Smoother boundary (underfitting) — considers too many points
- **K=n (all samples):** Always predicts majority class

**How to choose:**
- Use cross-validation and pick K with lowest validation error
- Rule of thumb: `K = sqrt(n)` where n = number of training samples
- Try odd K for binary classification (avoids ties)
- K should be small relative to n

---

### Q4. What distance metrics does KNN use?

**A:**

| Metric | Formula | Best For |
|---|---|---|
| Euclidean | `√Σ(xᵢ−yᵢ)²` | Continuous, isotropic data |
| Manhattan | `Σ\|xᵢ−yᵢ\|` | Grid-like data, outliers |
| Minkowski | `(Σ\|xᵢ−yᵢ\|^p)^(1/p)` | Generalization (p=1:Manhattan, p=2:Euclidean) |
| Cosine | `1 − (x·y)/(||x||·||y||)` | Text/high-dimensional sparse |
| Hamming | Count differing positions | Categorical, binary features |

---

### Q5. What is Euclidean distance?

**A:**
```
d(x, y) = √((x₁−y₁)² + (x₂−y₂)² + ... + (xₙ−yₙ)²)
```
The straight-line distance between two points in n-dimensional space. Most common default in KNN. Sensitive to scale → must standardize features.

---

### Q6. What is Manhattan distance?

**A:**
```
d(x, y) = |x₁−y₁| + |x₂−y₂| + ... + |xₙ−yₙ|
```
Sum of absolute differences — like walking city blocks. Less sensitive to outliers than Euclidean (no squaring). Preferred when features have very different scales or outliers are present.

---

### Q7. What is the curse of dimensionality?

**A:** In high dimensions, distances between points become nearly equal — all points are approximately the same distance from any query point. This makes KNN ineffective because "nearest" neighbors are no longer meaningfully close.

**Why it happens:** In high-D space, volume concentrates near the edges of the space. Most of the data lies in a shell near the boundary, not the center.

**Effect:** KNN accuracy degrades as dimensions increase beyond ~20–30 meaningful features.

**Solutions:** Dimensionality reduction (PCA, UMAP), feature selection, use approximate nearest neighbor methods.

---

### Q8. Is KNN a lazy learner?

**A:** Yes. KNN is a **lazy learner** (instance-based learning) because:
- No explicit training phase — just stores training data
- All computation happens at prediction time
- "Learning" = memorizing the training data

Opposite of "eager learners" (Logistic Regression, SVM, Neural Networks) that build a model during training.

**Trade-off:** Training is O(1) (trivial), but prediction is O(n × d) for each query — slow for large datasets.

---

### Q9. What is the difference between KNN and K-Means?

**A:**
| | KNN | K-Means |
|---|---|---|
| Type | Supervised (Classification/Regression) | Unsupervised (Clustering) |
| K meaning | Number of neighbors | Number of clusters |
| Training | None (lazy) | Iterative (EM-like) |
| Goal | Predict label | Group similar points |
| Requires labels | Yes | No |
| New point | Find k nearest training points | Assign to nearest centroid |

---

### Q10. What are the advantages of KNN?

**A:**
1. **Simple** — easy to understand and implement
2. **No training** — instant "learning"
3. **Non-parametric** — no assumptions about data distribution
4. **Naturally handles multi-class** — majority vote
5. **Adapts to local patterns** — different decision boundaries in different regions
6. **Works with any distance metric** — flexible for different data types

---

### Q11. What are the disadvantages of KNN?

**A:**
1. **Slow prediction** — O(n × d) per query, impractical for large n
2. **Memory intensive** — stores all training data
3. **Curse of dimensionality** — degrades in high dimensions
4. **Feature scaling required** — distance-sensitive
5. **No model interpretability** — hard to explain predictions
6. **Imbalanced classes** — biased toward majority class
7. **Sensitive to irrelevant features** — noise hurts distance calculations

---

### Q12. How does feature scaling affect KNN?

**A:** KNN relies entirely on distances. If one feature has a much larger scale (e.g., income in $100,000s vs age in 0–100), it dominates the distance calculation and other features become irrelevant.

**Fix: Always standardize features before KNN:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### Q13. What is weighted KNN?

**A:** In weighted KNN, closer neighbors have more influence than farther ones:
```
weight(i) = 1 / distance(xᵢ, query)
```
Prediction = weighted vote (classification) or weighted average (regression).

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

Weighted KNN handles ties better and is usually more accurate than uniform KNN.

---

### Q14. What is the time complexity of KNN?

**A:**
- **Training:** O(1) — just store data
- **Prediction (brute force):** O(n × d) per query — compute distance to all n points with d features
- **With KD-tree:** O(d × log n) per query (works well when d < 20)
- **With Ball-tree:** O(d × log n) per query (works better than KD-tree for high d)

For large datasets, use Approximate Nearest Neighbor (ANN) methods: FAISS, HNSW, ScaNN.

---

### Q15. What is a KD-Tree?

**A:** KD-tree (K-Dimensional Tree) is a space-partitioning data structure that organizes points in K-dimensional space for faster nearest neighbor search:
- Splits space along alternating dimensions at each level
- Allows pruning of search space → O(log n) for low dimensions
- Effective when d < 20

```python
KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
```

Breaks down for high dimensions — curse of dimensionality affects tree efficiency.

---

### Q16. What is Ball Tree?

**A:** Ball Tree organizes data into nested hyperspheres (balls):
- Builds a tree of nested balls of decreasing radius
- More effective than KD-tree for high-dimensional data (d > 20)
- Uses the triangle inequality to prune search space

```python
KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
```

---

### Q17. Can KNN be used for regression?

**A:** Yes — **KNN Regression** predicts the mean (or weighted mean) of the k nearest neighbors' target values:

```python
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)
```

Works well when the target is locally smooth. Poor performance for extrapolation or complex global patterns.

---

### Q18. How does KNN perform with imbalanced data?

**A:** Poorly. If 90% of training data is class A and 10% is class B, most neighbors of any query point will be class A → model predicts A almost always.

**Fix:**
1. Use weighted KNN with class weights
2. Oversample minority class (SMOTE) before training
3. Undersample majority class
4. Adjust decision threshold based on class priors

---

### Q19. How do you handle categorical features in KNN?

**A:** KNN needs numerical distances. For categorical features:
- **Binary/ordinal:** Encode as integers, include in Euclidean distance
- **Nominal (no order):** Use Hamming distance (count of mismatches) or one-hot encode
- **Mixed:** Use Gower distance (handles mix of categorical and numerical)

```python
# Hamming distance for categorical data
KNeighborsClassifier(n_neighbors=5, metric='hamming')
```

---

### Q20. What are Approximate Nearest Neighbor (ANN) methods?

**A:** For large-scale KNN (millions of vectors), exact KNN is too slow. ANN methods trade small accuracy loss for massive speed gains:

| Method | Approach | Best For |
|---|---|---|
| FAISS | IVF indexing, quantization | Dense vectors (embeddings) |
| HNSW | Hierarchical graph | High recall, fast query |
| ScaNN | Anisotropic quantization | Google's production search |
| LSH | Hash-based bucketing | Simple, large scale |
| Annoy | Random projection forests | Memory-mapped (read-only) |

```python
import faiss
index = faiss.IndexFlatL2(d)
index.add(X_train.astype('float32'))
D, I = index.search(X_query.astype('float32'), k=5)
```

---

## Quick Code Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[170,65,25],[165,80,30],[180,75,28],[160,85,45],
              [175,70,32],[155,90,50],[185,78,27],[162,88,48]])
y = np.array([1,0,1,0,1,0,1,0])  # 1=High, 0=Low

# Scale features (critical for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k via cross-validation
k_scores = []
for k in range(1, 8):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_scaled, y, cv=3, scoring='accuracy')
    k_scores.append(scores.mean())

best_k = np.argmax(k_scores) + 1
print(f"Best k: {best_k}, Accuracy: {max(k_scores):.3f}")

# Final model
knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn.fit(X_scaled, y)

# Predict new person
new_person = scaler.transform([[172, 72, 29]])
print("Prediction:", "High" if knn.predict(new_person)[0] == 1 else "Low")
print("Probability:", knn.predict_proba(new_person))
```
