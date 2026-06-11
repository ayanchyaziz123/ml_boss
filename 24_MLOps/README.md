# MLOps & Deployment — Interview Questions & Answers

---

## Small Dataset / System Example

```
Fraud Detection System — Production Scenario:

Training data: 1M transactions (Jan 2023 – Dec 2023)
Model: XGBoost, AUC=0.94 on test set

Deployed: Jan 2024
Monitored metrics:
  Week 1:  AUC=0.93, Precision=0.91, Recall=0.88  ✓
  Week 4:  AUC=0.90, Precision=0.87, Recall=0.85  ⚠️
  Week 8:  AUC=0.82, Precision=0.75, Recall=0.79  ❌
  Week 12: AUC=0.71  ← Model degradation
```

### Dataset Questions

**Q: The fraud detection model dropped from AUC=0.94 to AUC=0.71 over 12 weeks. What could cause this?**
> **A:** Two main causes: (1) **Data drift** — transaction patterns changed (new payment methods, seasonal patterns, geographic shifts); (2) **Concept drift** — fraudsters adapted their behavior in response to the model's detection patterns. The relationship between features and fraud changed. Both require retraining with recent data.

**Q: How would you detect data drift before AUC degrades?**
> **A:** Monitor the input feature distributions continuously: (1) **PSI (Population Stability Index)** for feature drift, (2) **KS test** for distribution shift, (3) **KL divergence** between training and production distributions, (4) Feature mean/std deviation tracking. Alert when PSI > 0.2 or KS test p-value < 0.05 for key features.

**Q: The system must serve 10,000 fraud checks per second with < 50ms latency. What infrastructure would you use?**
> **A:** (1) Containerize with Docker, deploy on Kubernetes for auto-scaling; (2) Use a feature store for pre-computed features (avoid recomputing at inference); (3) Model served via TorchServe/TF Serving/FastAPI with async inference; (4) Cache frequent feature lookups in Redis; (5) Load balance across multiple model replicas. XGBoost inference is fast (~1ms per prediction) — the bottleneck is usually feature retrieval.

---

## Questions & Answers

---

### Q1. What is MLOps?

**A:** MLOps (Machine Learning Operations) is a set of practices that combines ML, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and at scale.

Key components:
- **Data pipelines:** Automated data ingestion, validation, and transformation
- **Training pipelines:** Reproducible model training and experimentation
- **Model registry:** Version and track trained models
- **Serving infrastructure:** Deploy models as APIs/batch jobs
- **Monitoring:** Track model and data health in production

---

### Q2. What is model deployment?

**A:** Model deployment makes a trained model available for use in production. Types:

- **REST API (Real-time):** Model wrapped as HTTP endpoint, serves one prediction at a time
- **Batch inference:** Score large datasets on a schedule (offline)
- **Streaming:** Process real-time data streams (Kafka + model)
- **Edge deployment:** Run model on device (mobile, IoT) — no server needed

```python
# FastAPI deployment example
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: dict):
    return {"prediction": model.predict([list(features.values())])[0]}
```

---

### Q3. What is a REST API and why is it used for ML serving?

**A:** A REST API is a web service that accepts HTTP requests and returns responses (typically JSON). For ML serving:
- **Request:** JSON with input features
- **Response:** JSON with prediction and probability

```bash
curl -X POST "http://model-api/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 0.5, "feature2": 1.2}'
```

REST is used because it's language-agnostic, scalable, stateless, and easy to integrate with any application.

---

### Q4. What is Docker in ML context?

**A:** Docker packages the model, code, dependencies, and environment into a portable **container**:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.pkl .
COPY app.py .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Benefits: "Works on my machine" problem solved — same behavior everywhere (dev, staging, production).

---

### Q5. What is Kubernetes in ML context?

**A:** Kubernetes (K8s) orchestrates Docker containers at scale:
- **Auto-scaling:** Spin up more model replicas when traffic increases
- **Load balancing:** Distribute requests across replicas
- **Health checks:** Restart failed containers automatically
- **Rolling updates:** Deploy new model versions without downtime

In ML: Used to manage model serving pods, training jobs, and feature computation services.

---

### Q6. What is model monitoring?

**A:** Model monitoring tracks model health in production:

1. **Performance monitoring:** Track accuracy, AUC, precision, recall over time
2. **Data drift monitoring:** Track input feature distributions
3. **Prediction drift:** Track output distribution changes
4. **Infrastructure monitoring:** Latency, throughput, error rates

Tools: Evidently AI, Arize, WhyLogs, Prometheus + Grafana, MLflow.

---

### Q7. What is data drift?

**A:** Data drift (covariate shift) occurs when the distribution of input features in production differs from the training distribution:

```
Training: P_train(X) ≠ Production: P_prod(X)
```

Examples:
- Seasonal patterns: model trained on summer data, deployed in winter
- New user segments that weren't in training data
- Data collection changes (new sensor, new form fields)

**Detection:** PSI, KS test, Jensen-Shannon divergence on feature distributions.

---

### Q8. What is concept drift?

**A:** Concept drift occurs when the relationship between features and the target changes:
```
P(Y|X) changes over time
```

Even if features look the same, the model's predictions become wrong because the real-world pattern changed.

Examples:
- Fraudsters adapt to detection → fraud patterns change
- Economic recession changes customer behavior
- COVID changed many ML models' assumptions

**Detection:** Monitor model performance metrics (AUC, accuracy). Concept drift usually detected via degrading performance, not just feature distributions.

---

### Q9. What is A/B testing in ML?

**A:** A/B testing compares two model versions by serving them to different user groups simultaneously:

- **Group A (50% traffic):** Current model (control)
- **Group B (50% traffic):** New model (treatment)

Measure business metrics: click-through rate, conversion, revenue.
Test statistical significance before declaring a winner.

```python
# Check if improvement is statistically significant
from scipy.stats import chi2_contingency
contingency_table = [[model_a_conversions, model_a_non_conversions],
                      [model_b_conversions, model_b_non_conversions]]
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

---

### Q10. What is shadow deployment?

**A:** Shadow deployment runs a new model in parallel with the current model, but only the current model's predictions are used. The new model's predictions are logged but not shown to users.

**Purpose:**
- Validate new model on real traffic without risk
- Compare predictions: where do models disagree?
- Catch bugs or unexpected behavior before going live
- Measure performance without affecting users

---

### Q11. What is CI/CD in ML (Continuous Integration / Continuous Delivery)?

**A:**
- **CI (Continuous Integration):** Automatically test code changes — unit tests, integration tests, data validation
- **CD (Continuous Delivery):** Automatically deploy validated models to staging/production
- **CT (Continuous Training):** Automatically retrain models when data or performance changes

ML CI/CD pipeline:
```
Code commit → Tests → Data validation → Model training → 
Evaluation → Model registry → Deployment → Monitoring → 
Trigger retraining if drift detected
```

---

### Q12. What is MLflow?

**A:** MLflow is an open-source platform for managing the ML lifecycle:

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("auc", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
    mlflow.log_metric("accuracy", accuracy_score(y_val, model.predict(X_val)))
    
    # Save model
    mlflow.sklearn.log_model(model, "model")
```

Features: Experiment tracking, model registry, project packaging, model serving.

---

### Q13. What is model serialization?

**A:** Model serialization saves a trained model to disk:

```python
import joblib
import pickle

# joblib (preferred for sklearn — handles large numpy arrays efficiently)
joblib.dump(model, 'model.joblib')
model = joblib.load('model.joblib')

# pickle (general Python objects)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# ONNX (cross-platform)
import skl2onnx
onnx_model = skl2onnx.convert_sklearn(model, initial_types=...)

# PyTorch
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))
```

---

### Q14. What is model quantization?

**A:** Quantization reduces model size and speeds up inference by using lower-precision numbers:

- **FP32 → INT8:** 4× smaller, ~2–4× faster inference
- **FP32 → FP16:** 2× smaller, faster on GPU (tensor cores)

Types:
- **Post-Training Quantization (PTQ):** Quantize after training — easy, slight accuracy drop
- **Quantization-Aware Training (QAT):** Simulate quantization during training — better accuracy

```python
# PyTorch post-training quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

---

### Q15. What is model latency vs throughput?

**A:**
- **Latency:** Time to process one request (ms) — critical for real-time systems
- **Throughput:** Number of requests processed per second (QPS) — critical for batch/high-volume

Tradeoff:
- Batching increases throughput but increases latency per request
- Smaller model: lower latency, possibly lower accuracy
- More replicas: higher throughput, higher cost

```
Latency P50 (median), P95, P99 — tail latencies matter for SLAs
```

---

### Q16. What is a feature store?

**A:** A feature store is a centralized repository for storing, sharing, and serving ML features:

**Components:**
- **Offline store (historical):** Data warehouse for training — batch reads
- **Online store (real-time):** Key-value store for serving — low-latency lookup

**Benefits:**
- Share features across multiple models/teams
- Ensure consistency between training and serving (same features)
- Avoid recalculating expensive features

Tools: Feast, Hopsworks, Tecton, Databricks Feature Store.

---

### Q17. What is ONNX?

**A:** ONNX (Open Neural Network Exchange) is an open format for representing ML models across different frameworks:

```python
# Export from PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})

# Run with ONNX Runtime (fast, cross-platform inference)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

Train in PyTorch/TensorFlow → export to ONNX → deploy anywhere with ONNX Runtime.

---

### Q18. What is model drift and how do you trigger retraining?

**A:** Model drift = degradation of model performance over time due to data drift or concept drift.

**Retraining triggers:**
1. **Scheduled:** Retrain weekly/monthly regardless of performance
2. **Performance-based:** Retrain when AUC drops below threshold (e.g., < 0.85)
3. **Drift-based:** Retrain when PSI > 0.2 or KS test p < 0.05
4. **Data-volume-based:** Retrain after N new labeled samples collected

Automated retraining pipeline:
```
Monitor → Alert → Trigger retraining → Evaluate → 
Compare with current model → Deploy if better
```

---

### Q19. What is model versioning?

**A:** Model versioning tracks different trained model versions so you can:
- Roll back to previous version if new model fails
- Compare performance across versions
- Audit which model made which predictions
- Reproduce any past model

```python
# MLflow Model Registry
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="FraudDetector",
    version=3,
    stage="Production"  # Staging, Production, Archived
)
```

---

### Q20. What is model latency optimization?

**A:** Techniques to reduce model inference latency:

1. **Model quantization:** INT8/FP16 → 2–4× speedup
2. **Model pruning:** Remove unimportant weights → smaller model
3. **Knowledge distillation:** Train small student model to mimic large teacher
4. **ONNX Runtime:** Optimized inference engine
5. **TensorRT:** NVIDIA GPU-specific optimization
6. **Batching:** Process multiple requests together (tradeoff: adds latency)
7. **Caching:** Cache predictions for repeated inputs
8. **Async inference:** Non-blocking prediction calls
9. **Feature pre-computation:** Compute features ahead of time in feature store
10. **Model compilation:** `torch.compile()` in PyTorch 2.0

---

## Quick Code Example (FastAPI Model Serving)

```python
# app.py - Production ML serving with FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import time

app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model at startup
model = joblib.load("fraud_model.joblib")
scaler = joblib.load("scaler.joblib")

logger = logging.getLogger(__name__)

class Transaction(BaseModel):
    amount: float
    merchant_category: int
    hour_of_day: int
    is_foreign: int
    num_transactions_last_hour: int

class Prediction(BaseModel):
    is_fraud: bool
    fraud_probability: float
    latency_ms: float

@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    start = time.time()
    try:
        features = np.array([[
            transaction.amount,
            transaction.merchant_category,
            transaction.hour_of_day,
            transaction.is_foreign,
            transaction.num_transactions_last_hour
        ]])
        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0][1]
        latency = (time.time() - start) * 1000

        logger.info(f"Prediction: {prob:.4f}, Latency: {latency:.2f}ms")
        return Prediction(
            is_fraud=bool(prob > 0.5),
            fraud_probability=float(prob),
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": "v1.2"}

# Run: uvicorn app:app --host 0.0.0.0 --port 8080
```
