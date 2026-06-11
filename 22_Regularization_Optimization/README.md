# Regularization & Optimization — Interview Questions & Answers

---

## Small Dataset Example

```
Training a neural network for regression (predicting house prices):

Training loss curve:
Epoch  10: Train=0.45, Val=0.47
Epoch  50: Train=0.20, Val=0.22
Epoch 100: Train=0.08, Val=0.14  ← Start of overfitting
Epoch 150: Train=0.03, Val=0.22
Epoch 200: Train=0.01, Val=0.31  ← Severe overfitting

Without regularization: Best val loss at epoch ~50-80
```

### Dataset Questions

**Q: Looking at the training curve above, when should early stopping trigger and why?**
> **A:** Early stopping should trigger around **epoch 100–120** — when validation loss starts increasing while training loss continues to decrease. With patience=20, training would stop around epoch 120. This saves the model from epoch ~100 which has the best validation performance.

**Q: You add L2 regularization (λ=0.01). How does this change the training curve?**
> **A:** L2 penalizes large weights, so training loss decreases more slowly (model can't overfit as aggressively). The gap between training and validation loss narrows. The optimal stopping point shifts to later epochs because regularization is doing some of the generalization work.

**Q: Learning rate is set to 0.1 and loss oscillates wildly without converging. What would you do?**
> **A:** Reduce learning rate to 0.01 or 0.001. Also try: learning rate warmup (start low, increase, then decay), use Adam optimizer (adaptive LR), or use a learning rate scheduler (cosine annealing, ReduceLROnPlateau).

---

## Questions & Answers

---

### Q1. What is regularization?

**A:** Regularization adds constraints or penalties to the model to prevent overfitting by reducing complexity:

- **Explicit regularization:** L1, L2 penalties, dropout, weight decay
- **Implicit regularization:** Early stopping, data augmentation, batch normalization, SGD noise

The goal: Reduce generalization gap (Train error ≈ Test error) while accepting slightly higher training error.

---

### Q2. What is L1 regularization (Lasso)?

**A:** L1 adds the sum of absolute weight values to the loss:
```
Cost = Loss + λ Σ|wᵢ|
```

Properties:
- Pushes some weights **exactly to zero** → feature selection
- Produces sparse models
- Gradient is ±λ (constant) → promotes sparsity
- Useful when you believe few features are truly important

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)  # alpha = λ
```

---

### Q3. What is L2 regularization (Ridge)?

**A:** L2 adds the sum of squared weight values to the loss:
```
Cost = Loss + λ Σwᵢ²
```

Properties:
- Shrinks weights **toward zero but never exactly zero**
- Gradient = 2λw (proportional to weight) → large weights penalized more
- Closed-form solution exists
- Works well when all features contribute

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # alpha = λ
```

---

### Q4. What is the difference between L1 and L2?

**A:**
| | L1 | L2 |
|---|---|---|
| Penalty | Σ\|wᵢ\| | Σwᵢ² |
| Sparsity | Yes (weights → 0) | No (weights → small but non-zero) |
| Feature selection | Yes | No |
| Robust to outliers | More | Less |
| Solution | No closed-form | Closed-form exists |
| Best for | Sparse features, feature selection | Many small correlated effects |
| Geometry | Diamond constraint | Circle constraint |

---

### Q5. What is Dropout?

**A:** Dropout randomly zeroes out neurons during each training forward pass with probability p:

```python
nn.Dropout(p=0.5)  # 50% neurons dropped each batch
```

**Mechanism:** At each step, a different random subset of neurons is active → forces network to learn redundant representations.

**At test time:** All neurons active, weights scaled by (1-p) to compensate (or use inverted dropout during training, which is standard).

Effective dropout rate for sequential layers: p₁ × p₂ (compound).

---

### Q6. What is early stopping?

**A:** Early stopping monitors validation performance and stops training when it stops improving:

```python
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# PyTorch equivalent
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pt')
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        break  # stop training
```

Free regularization — no hyperparameter tuning needed beyond `patience`.

---

### Q7. What is data augmentation?

**A:** Data augmentation creates new training examples by applying label-preserving transformations to existing data:

**Images:**
```python
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(15),
transforms.ColorJitter(brightness=0.2),
transforms.RandomCrop(224, padding=4)
```

**Text:** Synonym replacement, back-translation, random insertion/deletion

**Tabular:** Adding small Gaussian noise, SMOTE for imbalanced data

Effectively increases dataset size and reduces overfitting.

---

### Q8. What is weight decay?

**A:** Weight decay directly scales down weights each step, equivalent to L2 regularization for SGD:

```
w = w - α(∇L + λw) = (1 - αλ)w - α∇L
```

The `(1 - αλ)` factor decays weights slightly each step.

**Note:** Weight decay ≠ L2 regularization for **Adam**. AdamW adds weight decay correctly (decoupled from gradient scaling), unlike Adam with L2.

```python
# PyTorch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

---

### Q9. What is gradient descent?

**A:** Gradient descent minimizes the loss by iteratively moving parameters in the direction of steepest descent:
```
θ = θ - α × ∇_θ L(θ)
```

**Three variants:**
- **Batch GD:** Full dataset per update — slow, stable
- **SGD:** 1 sample per update — fast, noisy
- **Mini-batch GD:** Small batch (32–512) per update — best balance

Deep learning uses mini-batch GD almost exclusively.

---

### Q10. What is stochastic gradient descent (SGD)?

**A:** SGD updates parameters using only one (or a mini-batch) of samples at a time:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Advantages:** Much faster than batch GD; noise acts as implicit regularization; can escape local minima.

**Disadvantages:** Noisy updates, may oscillate, sensitive to learning rate.

---

### Q11. What is momentum?

**A:** Momentum accelerates SGD in the consistent gradient direction and dampens oscillations:

```
v_t = β × v_{t-1} + α × ∇L
θ = θ - v_t
```

- `β=0.9` means "90% of previous velocity + 10% of new gradient"
- Builds up speed in consistent direction
- Dampens oscillations in directions where gradient reverses

Intuition: Like a ball rolling down a hill — it builds up speed.

---

### Q12. What is AdaGrad?

**A:** AdaGrad adapts the learning rate per parameter — infrequent parameters get larger updates:

```
G_t = G_{t-1} + ∇L²   (accumulate squared gradients)
θ = θ - (α / √(G_t + ε)) × ∇L
```

**Problem:** Accumulated G_t only grows → learning rate eventually → 0. Training stops early.

Useful for sparse data (NLP), not deep networks (stalls).

---

### Q13. What is RMSProp?

**A:** RMSProp fixes AdaGrad's decaying learning rate using exponential moving average:

```
v_t = β × v_{t-1} + (1-β) × ∇L²   (exponential moving avg of squared gradients)
θ = θ - (α / √(v_t + ε)) × ∇L
```

- `β=0.9` (default) — older gradients are forgotten
- Learning rate doesn't decay to zero
- Works well for RNNs and non-stationary problems

---

### Q14. What is Adam optimizer?

**A:** Adam (Adaptive Moment Estimation) combines momentum (1st moment) and RMSProp (2nd moment):

```
m_t = β₁m_{t-1} + (1-β₁)∇L         # first moment (momentum)
v_t = β₂v_{t-1} + (1-β₂)∇L²        # second moment (RMSProp)
m̂_t = m_t/(1-β₁ᵗ)                  # bias correction
v̂_t = v_t/(1-β₂ᵗ)                  # bias correction
θ = θ - α × m̂_t / (√v̂_t + ε)
```

Defaults: `β₁=0.9, β₂=0.999, ε=1e-8, α=0.001`

---

### Q15. What is the difference between SGD and Adam?

**A:**
| | SGD + Momentum | Adam |
|---|---|---|
| Adaptivity | Fixed LR for all params | Per-parameter adaptive LR |
| Convergence | Slower | Faster |
| Generalization | Often better (flatter minima) | Sometimes worse (sharp minima) |
| Hyperparameters | LR + momentum | LR + β₁, β₂, ε |
| Common use | Computer vision, ResNet | NLP, Transformers |

Large-batch training and Transformers: Adam. Small batch CV: SGD + momentum often wins.

---

### Q16. What is learning rate scheduling?

**A:** Adjusting the learning rate during training:

```python
# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

# Cosine annealing (best for large models)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

# Reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)

# Warmup + cosine (Transformers)
scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=1000,
                                             num_training_steps=total_steps)
```

---

### Q17. What is gradient clipping?

**A:** Gradient clipping prevents exploding gradients by limiting gradient magnitude:

```python
# Clip by norm (preferred — preserves direction)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

Clip by norm: if ||g|| > max_norm, scale all gradients: g = g × (max_norm/||g||). Essential for RNNs and Transformers.

---

### Q18. What is label smoothing?

**A:** Label smoothing replaces hard 0/1 labels with soft targets:
```
y_smooth = y × (1 - ε) + ε / K
```
For ε=0.1, K=10: Instead of [0,0,1,0,...] → [0.01, 0.01, 0.91, 0.01,...]

**Benefits:**
- Prevents the model from becoming overconfident
- Improves calibration
- Acts as regularization
- Used in Inception, language models

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

### Q19. What is batch normalization vs layer normalization?

**A:**
| | Batch Normalization | Layer Normalization |
|---|---|---|
| Normalizes over | Batch dimension (per feature) | Feature dimension (per sample) |
| Batch size sensitivity | Needs large batch | Works with batch_size=1 |
| Training vs inference | Different behavior (uses running stats at test) | Same behavior always |
| Best for | CNNs, feedforward networks | Transformers, RNNs |
| Formula | Normalize each feature over the batch | Normalize each sample over features |

---

### Q20. What is the bias-variance tradeoff in the context of regularization?

**A:**
```
Total Error = Bias² + Variance + Irreducible Noise
```

- **High bias (underfitting):** Model too simple, high training and test error
  - Fix: More features, more complex model, less regularization
  
- **High variance (overfitting):** Model too complex, low training error but high test error
  - Fix: More data, regularization (L1/L2/dropout), simpler model

Regularization **increases bias** (model constrained) but **reduces variance** (less sensitive to training data). The tradeoff: find λ where total test error is minimized.

---

## Quick Code Example

```python
import torch
import torch.nn as nn
import numpy as np

class RegularizedNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Regularization comparison
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

model = RegularizedNet(10, 64, 2, dropout_rate=0.3)

# AdamW = Adam + decoupled weight decay (better than Adam+L2)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4  # L2 regularization
)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    val_loss = loss.item()  # use actual val set in practice
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 20 == 0:
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
```
