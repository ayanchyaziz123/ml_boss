# 5 End-to-End ML Projects — Covering All Topics

Each project is designed to cover multiple algorithms, concepts, and production skills from the full interview guide.

---

## Project 1 — Customer Churn Prediction Platform

**Covers:** Linear Regression, Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Feature Engineering, Evaluation Metrics, MLOps

### Problem Statement
Build an end-to-end churn prediction system for a telecom company that predicts which customers will cancel their subscription within the next 30 days.

### Dataset
```
customer_id | tenure | monthly_charges | total_charges | contract_type |
internet_service | num_support_calls | payment_method | churn (0/1)

~7,000 rows — use Telco Customer Churn dataset (Kaggle)
```

### Step-by-Step Tasks

**Step 1 — EDA & Statistics**
- Compute mean, median, std for numerical features
- Plot churn rate by contract type, internet service
- Correlation matrix — find top features correlated with churn
- Hypothesis test: Is monthly_charges significantly different for churned vs non-churned customers? (t-test)

**Step 2 — Feature Engineering**
- `tenure_group` — bin tenure into 0–12, 13–24, 25–48, 49+ months
- `charge_per_month_ratio` = total_charges / tenure
- One-hot encode: contract_type, internet_service, payment_method
- Target encode high-cardinality categories
- Handle missing values in total_charges (median imputation)

**Step 3 — Model Training (compare all)**
```python
models = {
    'Logistic Regression': LogisticRegression(C=0.1, class_weight='balanced'),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5),
    'Random Forest':       RandomForestClassifier(n_estimators=200, class_weight='balanced'),
    'XGBoost':             XGBClassifier(scale_pos_weight=3, max_depth=4),
    'LightGBM':            LGBMClassifier(num_leaves=31, class_weight='balanced'),
}
```

**Step 4 — Evaluation**
- AUC-ROC, AUC-PR, F1, confusion matrix for all models
- Calibration curve — check if probabilities are trustworthy
- SHAP values — explain top 5 features driving churn

**Step 5 — MLOps / Deployment**
- Save best model with MLflow (track all experiments)
- Wrap as FastAPI endpoint: POST /predict → returns churn probability
- Containerize with Docker
- Monitor: add PSI check for monthly_charges distribution drift

### Key Interview Questions This Covers
- Why AUC-PR over AUC-ROC for imbalanced churn data?
- Why does LightGBM outperform Logistic Regression here?
- How do SHAP values help the business team act on predictions?
- How would you retrain the model when concept drift is detected?

### GitHub Structure
```
churn-prediction/
├── data/            # raw + processed
├── notebooks/       # EDA, feature engineering, modeling
├── src/
│   ├── features.py  # feature engineering pipeline
│   ├── train.py     # model training + MLflow logging
│   ├── evaluate.py  # metrics + SHAP
│   └── api.py       # FastAPI serving
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Project 2 — Real-Time Fraud Detection System

**Covers:** Logistic Regression, XGBoost, Neural Networks, Evaluation Metrics, Feature Engineering, Clustering (anomaly), Statistics, MLOps, Regularization

### Problem Statement
Build a real-time fraud detection system for credit card transactions that flags suspicious transactions with < 100ms latency and high recall (catching fraud is critical).

### Dataset
```
transaction_id | amount | merchant_category | hour | day_of_week |
is_foreign | card_age_days | num_transactions_last_1h |
avg_amount_last_7d | fraud (0/1)

~284,000 rows — use Credit Card Fraud Detection dataset (Kaggle)
Class imbalance: 0.17% fraud (severe imbalance)
```

### Step-by-Step Tasks

**Step 1 — Imbalance Analysis**
- Visualize class distribution
- Baseline: "predict all normal" achieves 99.83% accuracy — why is this useless?
- Choose evaluation metric: AUC-PR (not accuracy, not AUC-ROC)

**Step 2 — Feature Engineering**
```python
# Velocity features (fraud often has burst patterns)
df['tx_velocity_1h']   = rolling_count(transaction_id, window='1H')
df['amount_ratio']     = amount / avg_amount_last_7d
df['is_night']         = hour.between(0, 6).astype(int)
df['amount_log']       = np.log1p(amount)
df['high_risk_hour']   = hour.isin([1,2,3,4]).astype(int)
```

**Step 3 — Handling Imbalance**
```python
# Method 1: class weights
XGBClassifier(scale_pos_weight=577)  # 99.83/0.17

# Method 2: SMOTE
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Method 3: threshold tuning
# Don't use 0.5 — optimize threshold on Precision-Recall curve
```

**Step 4 — Model Comparison**
- Logistic Regression (baseline)
- XGBoost with scale_pos_weight
- Neural Network (MLP with dropout, class weights)
- Isolation Forest (unsupervised anomaly detection — no labels needed)

**Step 5 — Production System**
```
Transaction → Feature extraction (< 10ms, from Redis feature store)
           → XGBoost model inference (< 5ms)
           → Score > threshold → Flag as fraud
           → Log prediction for monitoring
           → Weekly model retraining trigger if AUC-PR drops > 2%
```

**Step 6 — Statistical Validation**
- McNemar's test: Is XGBoost significantly better than Logistic Regression?
- Bootstrap confidence intervals for AUC-PR
- Check calibration: are predicted fraud probabilities reliable?

### Key Interview Questions This Covers
- Why is recall the primary metric for fraud detection?
- How do you choose the optimal decision threshold?
- What is the cost of a false positive vs false negative?
- How does Isolation Forest detect fraud without labels?

### GitHub Structure
```
fraud-detection/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_threshold_optimization.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   ├── monitor.py   # drift detection
│   └── serve.py     # FastAPI + Redis
├── tests/
│   └── test_features.py
└── docker-compose.yml  # API + Redis
```

---

## Project 3 — Sentiment Analysis & Text Classification Pipeline

**Covers:** NLP, Naive Bayes, Logistic Regression, Neural Networks (LSTM, Transformers/BERT), Embeddings, Evaluation Metrics, Feature Engineering, MLOps

### Problem Statement
Build a multi-class sentiment classifier for product reviews (Negative / Neutral / Positive) that evolves from classical ML to BERT, showing the full NLP progression.

### Dataset
```
review_id | product_category | review_text | rating (1-5) | sentiment

~50,000 rows — use Amazon Product Reviews dataset
Map: rating 1-2 → Negative, 3 → Neutral, 4-5 → Positive
```

### Step-by-Step Tasks

**Step 1 — Text Preprocessing**
```python
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)         # remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]   # remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]    # lemmatize
    return ' '.join(tokens)
```

**Step 2 — Classical ML Baseline**
```python
# TF-IDF + Naive Bayes (fastest baseline)
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
nb    = MultinomialNB(alpha=0.1)

# TF-IDF + Logistic Regression (stronger baseline)
lr = LogisticRegression(C=1.0, solver='saga', multi_class='multinomial')

# Evaluate with macro F1 (3 classes, care about each equally)
```

**Step 3 — Deep Learning (LSTM)**
```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden, batch_first=True,
                              bidirectional=True, dropout=0.3)
        self.fc    = nn.Linear(hidden*2, 3)  # 3 classes

    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)
```

**Step 4 — Transformer (BERT Fine-tuning)**
```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=3)

# Fine-tune with HuggingFace Trainer
trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

**Step 5 — Model Comparison Table**
```
Model               | Macro F1 | Training Time | Inference/sec
--------------------|----------|---------------|---------------
Naive Bayes+TF-IDF  |   0.71   |     5s        |   50,000
LogReg+TF-IDF       |   0.78   |    30s        |   30,000
Bi-LSTM             |   0.83   |   20min       |    2,000
BERT fine-tuned     |   0.91   |   45min(GPU)  |      200
```

**Step 6 — Production**
- Serve BERT via FastAPI with tokenization in the request handler
- Cache tokenizer at startup
- Add language detection: reject non-English reviews
- Monitor: track prediction distribution shift weekly

### Key Interview Questions This Covers
- Why does BERT outperform Bi-LSTM for sentiment?
- When would you choose Naive Bayes over BERT?
- How does BPE tokenization handle "unhappiness" vs "happiness"?
- How do you reduce BERT inference latency for production?

### GitHub Structure
```
sentiment-analysis/
├── data/
├── notebooks/
│   ├── 01_eda_text.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_lstm.ipynb
│   └── 04_bert_finetuning.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_classical.py
│   ├── train_lstm.py
│   ├── train_bert.py
│   └── api.py
└── model_cards/
    └── bert_sentiment.md
```

---

## Project 4 — Image Classification with CNN + Transfer Learning

**Covers:** CNN, Transfer Learning, Data Augmentation, Neural Networks, Regularization, Optimization, Evaluation Metrics, MLOps, Generative Models (optional)

### Problem Statement
Build a plant disease detection system from leaf images that classifies 15 disease categories — starting from a custom CNN and improving with transfer learning from ResNet50.

### Dataset
```
38 classes of plant diseases
~87,000 images (256×256 RGB)
Use: PlantVillage dataset (Kaggle / TensorFlow Datasets)

Classes: Tomato_Late_blight, Apple_Cedar_rust, Corn_Gray_leaf_spot, ...
```

### Step-by-Step Tasks

**Step 1 — Data Pipeline**
```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Step 2 — Custom CNN (Baseline)**
```python
class PlantCNN(nn.Module):
    def __init__(self, n_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))
```

**Step 3 — Transfer Learning (ResNet50)**
```python
from torchvision import models

# Phase 1: Feature extraction (freeze backbone)
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(2048, 38)
)
# Train only the new head for 5 epochs with lr=1e-3

# Phase 2: Fine-tuning (unfreeze last 2 blocks)
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
# Train with lower lr=1e-4 for 10 more epochs
```

**Step 4 — Training & Optimization**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixed precision training (faster on GPU)
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

**Step 5 — Evaluation & Explainability**
```python
# Per-class accuracy, confusion matrix (38×38)
# Top-5 accuracy

# Grad-CAM visualization
from pytorch_grad_cam import GradCAM
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
grayscale_cam = cam(input_tensor=img_tensor)
# Shows which pixels the model focused on
```

**Step 6 — Comparison Table**
```
Model                  | Top-1 Acc | Top-5 Acc | Params  | Train Time
-----------------------|-----------|-----------|---------|------------
Custom CNN (3 blocks)  |   78.4%   |   94.1%   |  400K   |  2h (GPU)
ResNet50 (feature ext) |   91.2%   |   98.3%   |  23M    |  30min
ResNet50 (fine-tuned)  |   97.1%   |   99.5%   |  23M    |  1.5h
EfficientNetB3         |   98.0%   |   99.7%   |  12M    |  2h
```

**Step 7 — MLOps**
- Save model in ONNX format for cross-platform deployment
- Quantize to INT8 for edge devices (Raspberry Pi deployment)
- Wrap as REST API: POST /classify → returns disease + confidence + Grad-CAM heatmap URL
- Monitor: track top-1 accuracy weekly on new labeled batches

### Key Interview Questions This Covers
- Why does ResNet50 outperform your custom CNN with less training data?
- What is the difference between feature extraction and fine-tuning?
- Why use label smoothing and cosine annealing together?
- How does Grad-CAM help validate the model is looking at leaves not background?

### GitHub Structure
```
plant-disease-detection/
├── data/
├── notebooks/
│   ├── 01_eda_images.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   └── 04_gradcam_analysis.ipynb
├── src/
│   ├── dataset.py
│   ├── models/
│   │   ├── custom_cnn.py
│   │   └── resnet_transfer.py
│   ├── train.py
│   ├── evaluate.py
│   └── api.py
├── Dockerfile
└── README.md
```

---

## Project 5 — End-to-End Recommendation System with LLM Enhancement

**Covers:** Clustering (K-Means), PCA, Embeddings, Neural Networks, Transformers, NLP, Reinforcement Learning (bandits), Feature Engineering, Statistics, MLOps, Generative AI

### Problem Statement
Build a movie recommendation system that evolves from collaborative filtering → content-based → neural embeddings → LLM-enhanced recommendations, with A/B testing to compare approaches.

### Dataset
```
movies.csv:    movie_id | title | genres | year | description
ratings.csv:   user_id  | movie_id | rating (1-5) | timestamp
users.csv:     user_id  | age | gender | occupation

~100,000 ratings, 1,682 movies, 943 users
Use: MovieLens 100K dataset (freely available)
```

### Step-by-Step Tasks

**Step 1 — EDA & Statistics**
```python
# Rating distribution
ratings['rating'].describe()  # mean, std, quartiles
# Plot: ratings over time (temporal patterns)
# Chi-square test: Is rating distribution different across genres?
# Spearman correlation: Does age correlate with genre preferences?
```

**Step 2 — Collaborative Filtering (Matrix Factorization)**
```python
# User-Item matrix: 943 users × 1682 movies (very sparse ~94% missing)
# SVD / PCA on the matrix
from scipy.sparse.linalg import svds

U, sigma, Vt = svds(user_item_matrix, k=50)  # 50 latent factors
# U: user embeddings (943 × 50)
# Vt: movie embeddings (50 × 1682)
predicted_ratings = U @ np.diag(sigma) @ Vt

# Evaluate: RMSE on held-out ratings, top-10 Hit Rate
```

**Step 3 — Content-Based Filtering**
```python
# TF-IDF on movie descriptions + genre one-hot
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
desc_features = tfidf.fit_transform(movies['description'])
genre_features = pd.get_dummies(movies['genres'].str.get_dummies('|'))

# Combine: content_features = [desc_features | genre_features]
# Cosine similarity between movies → find similar movies
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(content_features)
```

**Step 4 — Neural Collaborative Filtering**
```python
class NeuralCF(nn.Module):
    def __init__(self, n_users, n_movies, embed_dim=32):
        super().__init__()
        # Matrix Factorization path
        self.user_embed_mf = nn.Embedding(n_users, embed_dim)
        self.movie_embed_mf = nn.Embedding(n_movies, embed_dim)
        # MLP path
        self.user_embed_mlp = nn.Embedding(n_users, embed_dim)
        self.movie_embed_mlp = nn.Embedding(n_movies, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, movie_ids):
        mf = self.user_embed_mf(user_ids) * self.movie_embed_mf(movie_ids)
        mlp_in = torch.cat([self.user_embed_mlp(user_ids),
                             self.movie_embed_mlp(movie_ids)], dim=1)
        mlp_out = self.mlp(mlp_in)
        return torch.sigmoid(mf.sum(1, keepdim=True) + mlp_out)
```

**Step 5 — Clustering User Segments**
```python
# K-Means on user embedding vectors from Neural CF
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

user_embeddings = model.user_embed_mlp.weight.data.numpy()

# Reduce to 2D for visualization
pca = PCA(n_components=2)
user_2d = pca.fit_transform(user_embeddings)

# Find k with silhouette score
km = KMeans(n_clusters=5, random_state=42)
user_clusters = km.fit_predict(user_embeddings)

# Analyze clusters: what genres does each cluster prefer?
```

**Step 6 — LLM Enhancement (RAG-style)**
```python
# Generate natural language explanations using Claude API
import anthropic

client = anthropic.Anthropic()

def explain_recommendation(user_history, recommended_movie):
    prompt = f"""
    User has watched and rated highly: {user_history}
    We are recommending: {recommended_movie}
    In 2 sentences, explain why this recommendation makes sense for this user.
    """
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# Also: use sentence embeddings for semantic movie similarity
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-MiniLM-L6-v2')
movie_embeddings = sbert.encode(movies['description'].tolist())
```

**Step 7 — A/B Testing & Evaluation**
```python
# Offline evaluation
def ndcg_at_k(recommended, relevant, k=10):
    """Normalized Discounted Cumulative Gain"""
    dcg = sum([1/np.log2(i+2) for i,m in enumerate(recommended[:k]) if m in relevant])
    ideal = sum([1/np.log2(i+2) for i in range(min(len(relevant), k))])
    return dcg / ideal if ideal > 0 else 0

# Compare all models
models_eval = {
    'Random':          random_recommendations,
    'Popularity':      popularity_recommendations,
    'SVD':             svd_recommendations,
    'Content-Based':   content_recommendations,
    'Neural CF':       ncf_recommendations,
    'Hybrid (NCF+CB)': hybrid_recommendations,
}

for name, rec_fn in models_eval.items():
    hits = [hit_rate_at_k(rec_fn(u), test_ratings[u]) for u in test_users]
    ndcg = [ndcg_at_k(rec_fn(u), test_ratings[u]) for u in test_users]
    print(f"{name:20s}  HitRate@10={np.mean(hits):.3f}  NDCG@10={np.mean(ndcg):.3f}")
```

**Step 8 — Exploration with Contextual Bandit (RL)**
```python
# Thompson Sampling bandit for exploration/exploitation
class ThompsonSamplingBandit:
    def __init__(self, n_items):
        self.alpha = np.ones(n_items)   # successes + 1
        self.beta  = np.ones(n_items)   # failures + 1

    def recommend(self, n=10):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argsort(samples)[-n:][::-1]

    def update(self, item_id, reward):
        if reward > 0:
            self.alpha[item_id] += 1
        else:
            self.beta[item_id] += 1
```

**Step 9 — Full Evaluation Table**
```
Model                | Hit@10 | NDCG@10 | Coverage | Diversity
---------------------|--------|---------|----------|----------
Random               |  0.04  |  0.02   |  100%    |   High
Popularity           |  0.18  |  0.11   |   5%     |   Low
SVD (50 factors)     |  0.42  |  0.28   |  45%     |   Med
Content-Based        |  0.38  |  0.24   |  60%     |   Med
Neural CF            |  0.51  |  0.34   |  55%     |   Med
Hybrid               |  0.57  |  0.39   |  62%     |   High
Hybrid + LLM explain |  0.57  |  0.39   |  62%     |   High  ← best UX
```

**Step 10 — MLOps Pipeline**
```
Daily pipeline:
  New ratings collected → Feature store updated →
  Bandit model updated online (no retraining) →
  Weekly: retrain Neural CF →
  A/B test new model vs production (5% traffic) →
  Promote if NDCG improves by > 2% (statistical significance check)
```

### Key Interview Questions This Covers
- What is the cold start problem and how do you handle it?
- Why does Neural CF outperform pure matrix factorization?
- How does the contextual bandit balance exploration vs exploitation?
- What is NDCG and why is it better than accuracy for ranking?
- How would you scale this to 10 million users?
- Why do we need diversity as a metric alongside accuracy?

### GitHub Structure
```
movie-recommender/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_neural_cf.ipynb
│   ├── 05_user_clustering.ipynb
│   ├── 06_llm_explanations.ipynb
│   └── 07_ab_testing.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── svd_model.py
│   │   ├── content_model.py
│   │   ├── neural_cf.py
│   │   └── bandit.py
│   ├── evaluation/
│   │   ├── metrics.py      # NDCG, Hit Rate, Coverage
│   │   └── ab_test.py
│   └── api/
│       └── app.py          # FastAPI serving
├── tests/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Topics Covered per Project

| Topic | P1 Churn | P2 Fraud | P3 Sentiment | P4 Image | P5 Recommender |
|-------|:---:|:---:|:---:|:---:|:---:|
| Linear/Logistic Regression | ✅ | ✅ | ✅ | | |
| Decision Tree / RF / Boosting | ✅ | ✅ | | | |
| Naive Bayes | | | ✅ | | |
| SVM | | ✅ | | | |
| KNN | | | | | ✅ |
| Clustering (K-Means) | | | | | ✅ |
| PCA | | | | | ✅ |
| Neural Networks (MLP) | ✅ | ✅ | ✅ | ✅ | ✅ |
| CNN + Transfer Learning | | | | ✅ | |
| RNN / LSTM | | | ✅ | | |
| Transformers / BERT | | | ✅ | | ✅ |
| Generative Models / LLM | | | | | ✅ |
| NLP | | | ✅ | | ✅ |
| Reinforcement Learning (Bandit) | | | | | ✅ |
| Feature Engineering | ✅ | ✅ | ✅ | ✅ | ✅ |
| Evaluation Metrics | ✅ | ✅ | ✅ | ✅ | ✅ |
| Regularization / Optimization | ✅ | ✅ | ✅ | ✅ | ✅ |
| Statistics & Hypothesis Testing | ✅ | ✅ | | | ✅ |
| MLOps & Deployment | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Recommended Order of Completion

```
Beginner  → Project 1 (Churn)       — Classical ML, feature engineering, MLOps basics
          → Project 2 (Fraud)        — Imbalance, anomaly detection, production serving

Intermediate → Project 3 (Sentiment) — NLP progression: TF-IDF → LSTM → BERT
             → Project 4 (Image)      — CV progression: Custom CNN → ResNet → ONNX

Advanced  → Project 5 (Recommender) — Full-stack ML: CF + Neural + LLM + RL + A/B Testing
```

---

## GitHub Portfolio Tips

1. Each project should have a clear `README.md` with: Problem → Dataset → Approach → Results table
2. Include a `model_card.md` explaining what the model does and its limitations
3. Add Jupyter notebooks with visualizations — recruiters look at these
4. Include unit tests for feature engineering and model predictions
5. Add a live demo link (Hugging Face Spaces is free and easy for NLP/CV projects)
