# Naive Bayes — Interview Questions & Answers

---

## Small Dataset Example (Email Spam Classification)

| Email Contains "free" | Contains "win" | Contains "meeting" | Contains "report" | Spam? |
|----------------------|----------------|-------------------|-------------------|-------|
| Yes | Yes | No | No | Spam |
| No | No | Yes | Yes | Ham |
| Yes | Yes | No | No | Spam |
| No | No | Yes | No | Ham |
| Yes | No | No | No | Spam |
| No | No | No | Yes | Ham |
| Yes | Yes | Yes | No | Spam |
| No | No | Yes | Yes | Ham |

### Dataset Questions

**Q: New email contains "free" and "meeting" but not "win" or "report". Is it Spam or Ham?**
> **A:** Using Naive Bayes, compute:
> - P(Spam) = 4/8 = 0.5, P(Ham) = 4/8 = 0.5
> - P("free"=Yes | Spam) = 4/4 = 1.0, P("free"=Yes | Ham) = 0/4 = 0.0
> - With Laplace smoothing, P("free"=Yes | Ham) = (0+1)/(4+2) = 0.167
> - P(Spam | features) ∝ 0.5 × 1.0 × 0.5 × 0.25 × 0.25 = high
> - **Prediction: Spam** (word "free" is a very strong spam indicator here)

**Q: What is the "naive" assumption being made about the features above?**
> **A:** The naive assumption is that "free", "win", "meeting", and "report" are **conditionally independent** given the class. In reality, "free" and "win" often appear together in spam, making them correlated — but Naive Bayes ignores this. Despite this wrong assumption, it often works well in practice.

**Q: If a new word "lottery" never appeared in training, what happens?**
> **A:** P("lottery" | Spam) = 0, making P(Spam | email) = 0 — called the **zero-frequency problem**. Fix: apply **Laplace smoothing** which adds 1 to every count, ensuring no probability is exactly zero.

---

## Questions & Answers

---

### Q1. What is Naive Bayes?

**A:** Naive Bayes is a family of probabilistic classifiers based on applying Bayes' theorem with the "naive" assumption that features are conditionally independent given the class label.

```
P(class | features) ∝ P(class) × Π P(featureᵢ | class)
```

Despite the unrealistic independence assumption, Naive Bayes performs surprisingly well in practice, especially for text classification.

---

### Q2. Explain Bayes' Theorem.

**A:** Bayes' theorem relates conditional probabilities:
```
P(A | B) = P(B | A) × P(A) / P(B)
```

In classification context:
```
P(class | X) = P(X | class) × P(class) / P(X)
```
- `P(class | X)` = posterior (what we want)
- `P(X | class)` = likelihood (how likely is this data given the class)
- `P(class)` = prior (how common is this class)
- `P(X)` = evidence (constant, can be ignored for classification)

---

### Q3. Why is it called "naive"?

**A:** It's "naive" because it assumes all features are **conditionally independent** given the class:

```
P(x₁, x₂, ..., xₙ | class) = P(x₁|class) × P(x₂|class) × ... × P(xₙ|class)
```

This is rarely true in real data. Words in a document are correlated, medical symptoms are correlated, etc. The model is "naive" about these correlations — it ignores them completely.

---

### Q4. What are the types of Naive Bayes?

**A:**

| Type | Best For | Likelihood |
|---|---|---|
| **Gaussian NB** | Continuous numerical features | Gaussian (normal) distribution per class |
| **Multinomial NB** | Text classification (word counts/TF-IDF) | Multinomial distribution |
| **Bernoulli NB** | Binary features (word present/absent) | Bernoulli distribution |
| **Complement NB** | Imbalanced text datasets | Uses complement class statistics |
| **Categorical NB** | Categorical features | Categorical distribution |

---

### Q5. What is conditional probability?

**A:** Conditional probability is the probability of event A given that event B has occurred:
```
P(A | B) = P(A ∩ B) / P(B)
```

Example: P(Email is Spam | contains "free") = (emails that are spam AND contain "free") / (all emails containing "free")

---

### Q6. What are the advantages of Naive Bayes?

**A:**
1. **Very fast** — training O(nd), prediction O(dc) where n=samples, d=features, c=classes
2. **Works with small data** — needs few samples to estimate parameters
3. **Handles many features well** — especially text with thousands of features
4. **Naturally multi-class** — no need for OvR strategies
5. **Robust to irrelevant features** — averages out
6. **Interpretable** — can inspect learned probabilities
7. **Online learning** — can update with new data without retraining from scratch

---

### Q7. What are the limitations of Naive Bayes?

**A:**
1. **Independence assumption** — often violated, limits accuracy
2. **Zero-frequency problem** — unseen feature values → zero probability
3. **Poor probability estimates** — probabilities are often biased (too extreme: 0.99 or 0.01)
4. **Cannot capture feature interactions** — ignores correlations
5. **Gaussian assumption** may not hold for continuous features
6. **Sensitive to feature encoding** — how you represent features matters

---

### Q8. What are common text classification use cases?

**A:**
1. **Spam detection** — email spam vs ham
2. **Sentiment analysis** — positive/negative reviews
3. **News categorization** — sports/politics/tech
4. **Language detection** — identifying language of text
5. **Document classification** — legal, medical, financial docs
6. **Fake news detection**
7. **Intent classification** in chatbots

Naive Bayes is a strong baseline for all these tasks and is still used in production for its speed.

---

### Q9. How does Naive Bayes handle continuous data?

**A:** **Gaussian Naive Bayes:**
- Assume each feature follows a Gaussian distribution within each class
- Estimate mean (μ) and variance (σ²) for each feature-class combination
- Use Gaussian PDF for likelihood:
```
P(x | class) = (1/√(2πσ²)) × exp(−(x−μ)²/(2σ²))
```

Alternative: Discretize continuous features into bins and use Multinomial NB.

---

### Q10. When does Naive Bayes perform well?

**A:** Naive Bayes works best when:
1. **Independence assumption is approximately true** (rare in reality, but still works)
2. **High-dimensional, sparse data** — text documents
3. **Small training data** — quick to train reliably
4. **Multi-class problems** with many classes
5. **Real-time classification** needed (very fast inference)
6. **Strong baseline** needed quickly with minimal preprocessing

---

### Q11. What is Gaussian Naive Bayes?

**A:** Gaussian Naive Bayes assumes each feature follows a normal distribution within each class:

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

# Inspect learned parameters
print("Class means:", model.theta_)      # shape: (n_classes, n_features)
print("Class variances:", model.var_)   # shape: (n_classes, n_features)
```

Best for: continuous numerical features like measurements, sensor readings.

---

### Q12. What is Multinomial Naive Bayes?

**A:** Models the frequency of each word/feature in each class using a multinomial distribution. Best for word counts or TF-IDF vectors in text classification:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(text_data)

model = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
model.fit(X_counts, y)
```

---

### Q13. What is Bernoulli Naive Bayes?

**A:** Models whether each feature is present (1) or absent (0) — binary features. For text: whether a word appears in a document, not how many times.

```python
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB(alpha=1.0)
model.fit(X_binary, y)
```

Use when: document length doesn't matter, only word presence/absence.
Multinomial is better when word frequency matters.

---

### Q14. What is Laplace smoothing?

**A:** Laplace smoothing (additive smoothing) adds a small count α to every observation to prevent zero probabilities:

```
P(word | class) = (count(word, class) + α) / (total_words_in_class + α × vocab_size)
```

With α=1 (Laplace smoothing):
- Words that never appeared in a class get P = 1/(total + vocab_size) instead of 0
- Prevents "zero-frequency problem"

---

### Q15. What is the zero-frequency problem?

**A:** If a feature value never appeared in training for a particular class, its probability = 0. This makes the entire posterior probability 0 (due to multiplication), regardless of other features.

```
P("lottery" | Spam) = 0
→ P(Spam | email with "lottery") = 0 (wrong! email could still be spam)
```

**Fix:** Laplace smoothing (`alpha=1` in sklearn).

---

### Q16. Is Naive Bayes a generative or discriminative model?

**A:** Naive Bayes is a **generative** model — it models the joint probability distribution P(X, Y) = P(X|Y) × P(Y). From this, it derives P(Y|X) using Bayes' theorem.

Compare to **discriminative** models like Logistic Regression and SVM, which directly model P(Y|X) without modeling the data distribution.

Generative models can generate new data samples; discriminative models cannot.

---

### Q17. How does Naive Bayes scale with data?

**A:** Extremely well:
- **Training:** O(n × d) — just counting occurrences
- **Prediction:** O(d × c) — just multiplications
- **Memory:** O(d × c) — store one probability per feature per class

For 1 million documents with 50,000 word vocabulary: training in seconds.
Logistic Regression or SVM would require minutes/hours.

---

### Q18. Why does Naive Bayes give poor probability estimates?

**A:** The independence assumption leads to overconfident predictions. Because it multiplies many probabilities together (assuming independence), probabilities tend toward extreme values (near 0 or 1). The actual probabilities are biased — not well-calibrated.

**Fix for calibrated probabilities:** Apply isotonic regression or Platt scaling after training.

---

### Q19. What is the difference between Naive Bayes and Logistic Regression?

**A:**
| | Naive Bayes | Logistic Regression |
|---|---|---|
| Type | Generative | Discriminative |
| Assumption | Feature independence | No independence assumption |
| Small data | Better | May overfit |
| Large data | LR usually wins | Better accuracy |
| Speed | Very fast | Fast |
| Probability calibration | Poor | Better |
| Feature correlations | Ignored | Handles well |

---

### Q20. When does Naive Bayes outperform Logistic Regression?

**A:** Naive Bayes outperforms Logistic Regression when:
1. **Very small training data** — Naive Bayes is more data-efficient
2. **High-dimensional, sparse text** — independence assumption approximately holds
3. **Fast inference required** — Naive Bayes is orders of magnitude faster
4. **Features are truly independent** — rare but does happen
5. **Online/streaming learning** — easy to update counts incrementally

---

## Quick Code Example

```python
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Binary features (word presence in email)
X = np.array([[1,1,0,0],[0,0,1,1],[1,1,0,0],[0,0,1,0],
              [1,0,0,0],[0,0,0,1],[1,1,1,0],[0,0,1,1]])
y = np.array([1,0,1,0,1,0,1,0])  # 1=Spam, 0=Ham

# Bernoulli NB (binary features)
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X, y)
print("Bernoulli NB Accuracy:", accuracy_score(y, bnb.predict(X)))

# Predict new email: contains "free" and "meeting" but not "win" or "report"
new_email = np.array([[1, 0, 1, 0]])
print("Prediction (1=Spam):", bnb.predict(new_email))
print("Probability:", bnb.predict_proba(new_email))

# Log probabilities of features per class
print("\nLog P(feature=1 | Spam):", bnb.feature_log_prob_[1])
print("Log P(feature=1 | Ham):", bnb.feature_log_prob_[0])
```
