# Decision Tree — Interview Questions & Answers

---

## Small Dataset Example

| Weather | Temperature | Humidity | Windy | Play Tennis? |
|---------|-------------|----------|-------|--------------|
| Sunny | Hot | High | No | No |
| Sunny | Hot | High | Yes | No |
| Overcast | Hot | High | No | Yes |
| Rainy | Mild | High | No | Yes |
| Rainy | Cool | Normal | No | Yes |
| Rainy | Cool | Normal | Yes | No |
| Overcast | Cool | Normal | Yes | Yes |
| Sunny | Mild | High | No | No |
| Sunny | Cool | Normal | No | Yes |
| Rainy | Mild | Normal | No | Yes |

### Dataset Questions

**Q: What is the entropy of the target variable "Play Tennis"?**
> **A:** 6 Yes, 4 No out of 10. Entropy = −(6/10)log₂(6/10) − (4/10)log₂(4/10) = −0.6×(−0.737) − 0.4×(−1.322) = 0.442 + 0.529 = **0.971 bits**

**Q: Which feature would a Decision Tree likely split on first using Information Gain?**
> **A:** The feature with the highest information gain. In this classic dataset (Tennis), "Outlook/Weather" typically gives the highest information gain (~0.246 bits) and is chosen as the root node.

**Q: If max_depth=1, what does the tree look like?**
> **A:** The tree has only one split (the root node split on the best feature) and two leaf nodes. This is called a **decision stump**.

---

## Questions & Answers

---

### Q1. What is entropy?

**A:** Entropy measures the impurity or disorder in a set of labels:
```
Entropy(S) = −Σ pᵢ log₂(pᵢ)
```
- Entropy = 0 → perfectly pure (all same class)
- Entropy = 1 → maximum disorder (50/50 split in binary)

Example: {5 Yes, 5 No} → Entropy = −0.5 log₂(0.5) − 0.5 log₂(0.5) = **1.0**

---

### Q2. What is information gain?

**A:** Information Gain measures how much a feature reduces entropy after splitting:
```
IG(S, A) = Entropy(S) − Σ (|Sᵥ|/|S|) × Entropy(Sᵥ)
```
The feature with the **highest information gain** is chosen as the split at each node.
Used by the **ID3** algorithm.

---

### Q3. What is Gini impurity?

**A:** Gini impurity measures the probability of incorrectly classifying a randomly chosen element:
```
Gini(S) = 1 − Σ pᵢ²
```
- Gini = 0 → perfectly pure node
- Gini = 0.5 → maximum impurity (binary case)

Used by **CART** algorithm (sklearn's default).

**Gini vs Entropy:** Both give similar results. Gini is computationally faster (no log calculation). Entropy tends to create more balanced trees.

---

### Q4. How does a Decision Tree work?

**A:** Step by step:
1. Start at the root with all training data
2. Find the feature and split point that best separates classes (highest IG or lowest Gini)
3. Split data into subsets based on that feature
4. Repeat recursively for each subset (child node)
5. Stop when: node is pure, max depth reached, min samples threshold hit, or no improvement possible
6. Leaf nodes contain the majority class (classification) or mean value (regression)

---

### Q5. What is pruning?

**A:** Pruning reduces tree size to prevent overfitting.

- **Pre-pruning (early stopping):** Stop growing the tree early using:
  - `max_depth`: limit tree depth
  - `min_samples_split`: minimum samples needed to split a node
  - `min_samples_leaf`: minimum samples in a leaf
  
- **Post-pruning:** Grow the full tree, then remove branches that don't improve validation performance.
  - `ccp_alpha` in sklearn (cost-complexity pruning)

---

### Q6. How do Decision Trees overfit?

**A:** A fully grown tree memorizes training data by creating very specific rules for each sample. Signs of overfitting:
- Very deep tree with many leaf nodes
- Training accuracy = 100%, test accuracy much lower
- Leaf nodes contain very few samples

**Fix:** Prune the tree using `max_depth`, `min_samples_leaf`, or `ccp_alpha`.

---

### Q7. What are the advantages of Decision Trees?

**A:**
1. Easy to understand and visualize
2. No feature scaling needed
3. Handles both numerical and categorical features
4. Handles missing values
5. Captures non-linear relationships
6. Fast training and prediction
7. Feature importance built-in
8. No assumptions about data distribution

---

### Q8. What are the disadvantages of Decision Trees?

**A:**
1. **Overfitting** — easily memorizes training data
2. **High variance** — small data changes can produce very different trees
3. **Biased toward features with many categories**
4. **Cannot extrapolate** beyond training data range (regression)
5. **Unstable** — not robust to noisy data
6. **Greedy** — locally optimal splits may not be globally optimal

---

### Q9. How are splits chosen?

**A:** At each node:
1. For each feature, sort unique values
2. Try each unique value as a split point
3. Calculate impurity (Gini) or information gain for each split
4. Choose the feature + split point that maximizes information gain (or minimizes Gini)

For continuous features: try midpoints between consecutive sorted values.
For categorical features: try each category or subset of categories.

---

### Q10. Can Decision Trees handle missing values?

**A:** Yes, using **surrogate splits** — when a feature value is missing, use the feature most correlated with the primary split as a substitute.

Sklearn by default does not handle missing values natively (requires imputation). XGBoost and LightGBM natively handle missing values in tree splits.

---

### Q11. What is a leaf node?

**A:** A leaf node (terminal node) is the final node in a decision tree — it contains no further splits. The leaf stores:
- **Classification:** majority class of training samples in that leaf
- **Regression:** mean value of training samples in that leaf

---

### Q12. What is a root node?

**A:** The root node is the top-most node in the tree. It contains all training samples and represents the first (most important) split — the feature with the highest information gain across the entire dataset.

---

### Q13. What is tree depth?

**A:** Tree depth is the length of the longest path from the root node to a leaf node. Depth 0 = just the root (no splits). Each level of splits increases depth by 1. Deeper trees are more complex and more prone to overfitting.

---

### Q14. What is CART?

**A:** CART (Classification and Regression Trees) is the algorithm used by sklearn. It:
- Uses **Gini impurity** for classification, **MSE** for regression
- Produces **binary splits** (always two children)
- Can build regression trees
- Forms the base for Random Forest and Gradient Boosting

---

### Q15. What is ID3?

**A:** ID3 (Iterative Dichotomiser 3) is one of the earliest tree algorithms:
- Uses **Information Gain (Entropy)** as split criterion
- Handles only **categorical** features
- Produces **multi-way splits** (one branch per category value)
- Biased toward features with many unique values

---

### Q16. What is the difference between classification and regression trees?

**A:**
| | Classification Tree | Regression Tree |
|---|---|---|
| Target | Categorical (classes) | Continuous (numbers) |
| Split criterion | Gini / Entropy | MSE / MAE |
| Leaf output | Majority class | Mean value |
| Evaluation | Accuracy, F1 | RMSE, R² |

---

### Q17. What is max_depth hyperparameter?

**A:** `max_depth` limits how deep the tree can grow. Shallower trees are simpler but may underfit. Deeper trees can overfit.

Typical tuning: start with `max_depth=3` or `4`, tune via cross-validation.

---

### Q18. What is min_samples_split?

**A:** `min_samples_split` specifies the minimum number of samples a node must have before it can be split. Higher values create simpler, more generalized trees. Default in sklearn = 2.

---

### Q19. How do you visualize a Decision Tree?

**A:**
```python
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=feature_names,
          class_names=['No', 'Yes'], filled=True)
plt.show()

# Text representation
print(export_text(model, feature_names=feature_names))
```

---

### Q20. What is the relationship between Decision Tree depth and bias-variance?

**A:**
- **Shallow tree** (depth 1–3): High bias, low variance → underfits
- **Deep tree** (depth 10+): Low bias, high variance → overfits
- **Optimal depth:** Found via cross-validation — the sweet spot where test error is minimized

---

## Quick Code Example

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Weather dataset (encoded)
X = [[0,0,0,0],[0,0,0,1],[1,0,0,0],[2,1,0,0],[2,2,1,0],
     [2,2,1,1],[1,2,1,1],[0,1,0,0],[0,2,1,0],[2,1,1,0]]
y = [0,0,1,1,1,0,1,0,1,1]

model = DecisionTreeClassifier(max_depth=3, criterion='gini')
model.fit(X, y)

print("Accuracy:", accuracy_score(y, model.predict(X)))
print("Feature Importances:", model.feature_importances_)

plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=['Weather','Temp','Humidity','Windy'],
          class_names=['No','Yes'], filled=True)
plt.show()
```
