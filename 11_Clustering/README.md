# Clustering — Interview Questions & Answers

---

## Small Dataset Example

| Customer ID | Annual Spend ($k) | Visit Frequency | Avg Basket ($) |
|-------------|-------------------|-----------------|----------------|
| 1 | 2.1 | 5 | 42 |
| 2 | 2.3 | 4 | 45 |
| 3 | 8.5 | 20 | 85 |
| 4 | 9.0 | 22 | 90 |
| 5 | 15.0 | 35 | 150 |
| 6 | 14.5 | 33 | 145 |
| 7 | 2.0 | 6 | 40 |
| 8 | 8.8 | 21 | 88 |

### Dataset Questions

**Q: If you run K-Means with k=3, what clusters would you expect?**
> **A:** Three natural groups: (1) Low spenders: rows 1,2,7 (~$2k spend, ~5 visits), (2) Mid spenders: rows 3,4,8 (~$8.7k spend, ~21 visits), (3) High spenders: rows 5,6 (~$14.8k spend, ~34 visits). These correspond to budget, regular, and premium customer segments.

**Q: Why must you scale features before running K-Means on this dataset?**
> **A:** Annual Spend ranges 2–15 (13-unit range), Visit Frequency ranges 4–35 (31-unit range), Avg Basket ranges 40–150 (110-unit range). Without scaling, Avg Basket dominates the distance calculation. Standardize all features first so each has equal weight in clustering.

**Q: How would you choose k=3 if you didn't know the answer in advance?**
> **A:** Use the Elbow Method: plot inertia (within-cluster sum of squares) vs k. The "elbow" where inertia stops dropping sharply indicates the optimal k. Also use Silhouette Score: pick k that maximizes the average silhouette width (closer to 1 = better separation).

---

## Questions & Answers

---

### Q1. What is clustering?

**A:** Clustering is an unsupervised learning task that groups similar data points together without using labeled data. The goal is to find natural groupings where:
- Points within a cluster are similar to each other (intra-cluster cohesion)
- Points in different clusters are dissimilar (inter-cluster separation)

Applications: customer segmentation, document grouping, anomaly detection, image compression.

---

### Q2. What is K-Means?

**A:** K-Means is an iterative clustering algorithm that partitions data into exactly K clusters by minimizing within-cluster variance (inertia):

```
Objective: minimize Σₖ Σ(x ∈ Cₖ) ||x − μₖ||²
```

Where `μₖ` is the mean (centroid) of cluster k. It finds K centroids that minimize total squared distance from each point to its nearest centroid.

---

### Q3. How does K-Means work?

**A:** The K-Means algorithm:
1. **Initialize:** Randomly place K centroids (or use K-Means++)
2. **Assign:** Each point → nearest centroid (Euclidean distance)
3. **Update:** Recompute each centroid as the mean of its assigned points
4. **Repeat steps 2–3** until centroids stop moving (convergence)

Time complexity: O(n × K × d × iterations)

---

### Q4. What is inertia?

**A:** Inertia = total within-cluster sum of squared distances from each point to its cluster centroid:
```
Inertia = Σₖ Σ(x ∈ Cₖ) ||x − μₖ||²
```
- Lower inertia = tighter clusters (better)
- Inertia always decreases as K increases → use elbow method to balance

---

### Q5. What is the elbow method?

**A:** A heuristic for selecting optimal K:
1. Run K-Means for K = 1, 2, 3, ..., 10
2. Plot inertia vs K
3. Find the "elbow" — the point where adding more clusters gives diminishing returns

```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
plt.plot(range(1, 11), inertias, 'bx-')
plt.xlabel('K'); plt.ylabel('Inertia')
plt.title('Elbow Method')
```

Limitation: The "elbow" is often ambiguous. Combine with silhouette score.

---

### Q6. What is a centroid?

**A:** A centroid is the mean point of all samples assigned to a cluster:
```
μₖ = (1/|Cₖ|) Σ(x ∈ Cₖ) x
```
In K-Means, centroids are recalculated after each assignment step. The final centroids represent the "center" of each cluster. They may not correspond to any actual data point.

---

### Q7. What are the limitations of K-Means?

**A:**
1. **Must specify K** in advance — unknown in real problems
2. **Sensitive to initialization** — different random starts → different clusters
3. **Assumes spherical clusters** — fails for elongated or irregular shapes
4. **Sensitive to outliers** — outliers pull centroids toward them
5. **Assumes equal cluster sizes** — may split large, dense clusters
6. **Only finds local optima** — not guaranteed to find global optimum
7. **Requires feature scaling** — distance-based
8. **Fails for non-convex clusters** — e.g., rings or crescents

---

### Q8. What is Hierarchical Clustering?

**A:** Hierarchical clustering builds a tree of clusters (dendrogram) without requiring K in advance.

**Agglomerative (bottom-up):**
1. Start: each point is its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains

**Linkage criteria (how to measure cluster distance):**
- **Single:** min distance between points
- **Complete:** max distance between points
- **Average:** mean distance between all point pairs
- **Ward:** minimize within-cluster variance (like K-Means)

---

### Q9. What is DBSCAN?

**A:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters as dense regions separated by low-density regions.

**Key concepts:**
- **Core point:** Has at least `min_samples` points within radius `eps`
- **Border point:** Within eps of a core point but not a core itself
- **Noise point:** Not within eps of any core point (outlier)

**Advantages:** Finds arbitrarily shaped clusters, handles noise/outliers, no need to specify K.

---

### Q10. What is the difference between K-Means and DBSCAN?

**A:**
| | K-Means | DBSCAN |
|---|---|---|
| K required | Yes | No |
| Cluster shape | Spherical only | Arbitrary |
| Outlier handling | Forced into a cluster | Labeled as noise |
| Scales with data | O(n×K×d) — fast | O(n log n) with index |
| Parameters | K | eps, min_samples |
| Imbalanced clusters | Struggles | Handles well |
| Best for | Well-separated, round clusters | Arbitrary shapes, noise |

---

### Q11. What is the Silhouette Score?

**A:** Silhouette score measures how similar a point is to its own cluster vs other clusters:
```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```
- `a(i)` = mean distance to all points in same cluster
- `b(i)` = mean distance to all points in nearest other cluster
- Range: −1 to 1 (1 = well-clustered, 0 = on boundary, −1 = misassigned)

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, labels)
```

---

### Q12. What is K-Means++ initialization?

**A:** K-Means++ is a smarter initialization strategy that spreads initial centroids far apart:
1. Choose first centroid randomly
2. For each remaining centroid: choose the next point with probability proportional to its squared distance from the nearest existing centroid
3. Repeat until K centroids are chosen

**Result:** Faster convergence, better final clusters, less likely to get stuck in bad local optima. Default in sklearn (`init='k-means++'`).

---

### Q13. What is the difference between K-Means and Gaussian Mixture Models (GMM)?

**A:**
| | K-Means | GMM |
|---|---|---|
| Assignment | Hard (each point → one cluster) | Soft (probabilities) |
| Cluster shape | Spherical (equal variance) | Elliptical (full covariance) |
| Output | Cluster labels | Probability of each cluster |
| Algorithm | Expectation of distances | EM algorithm |
| Flexibility | Less | More |
| Interpretability | More | Less |

---

### Q14. What is Mini-Batch K-Means?

**A:** Mini-Batch K-Means speeds up standard K-Means on large datasets:
- Instead of using all data per iteration, uses random mini-batches
- Updates centroids incrementally using running averages
- ~3–5x faster than standard K-Means with minimal accuracy loss

```python
from sklearn.cluster import MiniBatchKMeans
mbkm = MiniBatchKMeans(n_clusters=3, batch_size=100, random_state=42)
```

---

### Q15. What is the Davies-Bouldin Index?

**A:** Davies-Bouldin Index measures cluster quality — lower is better:
```
DB = (1/K) Σₖ max_{j≠k} [(s_k + s_j) / d(μₖ, μⱼ)]
```
- `s_k` = average distance within cluster k
- `d(μₖ, μⱼ)` = distance between centroids

Lower DB = compact, well-separated clusters.

```python
from sklearn.metrics import davies_bouldin_score
score = davies_bouldin_score(X_scaled, labels)
```

---

### Q16. What is Agglomerative Clustering?

**A:** A hierarchical clustering method that starts with each point as its own cluster and merges the closest pair iteratively:

```python
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X_scaled)
```

No random initialization. Deterministic. Produces a full dendrogram. More expensive: O(n² log n).

---

### Q17. What is the WCSS (Within-Cluster Sum of Squares)?

**A:** WCSS = Inertia = sum of squared distances from each point to its cluster centroid. Same as inertia. Used in elbow method. Minimized by K-Means algorithm. Always decreases with more clusters.

---

### Q18. How does DBSCAN handle outliers?

**A:** DBSCAN explicitly labels outliers as **noise** (label = −1). Points that are not core points and not reachable from any core point are classified as noise. This is a key advantage: outliers don't distort cluster shapes (unlike K-Means where outliers are forced into a cluster and pull centroids).

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=3)
labels = db.fit_predict(X_scaled)
n_noise = (labels == -1).sum()
print(f"Noise points: {n_noise}")
```

---

### Q19. What are eps and min_samples in DBSCAN?

**A:**
- **eps (ε):** The radius defining the neighborhood of a point. Points within eps of each other are neighbors. Smaller eps → fewer core points, more noise.
- **min_samples:** Minimum number of points within eps to be a core point. Higher min_samples → denser clusters required, more noise.

**Tuning tip:** Plot the k-nearest neighbor distance graph (k = min_samples). Look for the "elbow" — that's the optimal eps.

---

### Q20. What is the difference between K-Means and K-Medoids?

**A:**
| | K-Means | K-Medoids (PAM) |
|---|---|---|
| Center | Mean of cluster | Actual data point (medoid) |
| Outlier sensitivity | High (mean pulled by outliers) | Low (medoid is a real point) |
| Distance metric | Must use Euclidean | Any distance metric |
| Speed | Faster | Slower |
| Use case | Clean numerical data | Data with outliers, non-Euclidean distances |

---

## Quick Code Example

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

X = np.array([[2.1,5,42],[2.3,4,45],[8.5,20,85],[9.0,22,90],
              [15.0,35,150],[14.5,33,145],[2.0,6,40],[8.8,21,88]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with elbow method
inertias, sil_scores = [], []
for k in range(2, 6):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

print("Silhouette scores:", sil_scores)
best_k = np.argmax(sil_scores) + 2
print(f"Best K: {best_k}")

# Final model
km = KMeans(n_clusters=3, init='k-means++', random_state=42)
labels = km.fit_predict(X_scaled)
print("Cluster labels:", labels)
print("Centroids (scaled):", km.cluster_centers_)

# DBSCAN
db = DBSCAN(eps=0.8, min_samples=2)
db_labels = db.fit_predict(X_scaled)
print("\nDBSCAN labels:", db_labels)
print("Noise points:", (db_labels == -1).sum())
```
