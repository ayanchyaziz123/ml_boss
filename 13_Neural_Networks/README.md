# Neural Networks (ANN / MLP) — Interview Questions & Answers

---

## Small Dataset Example

| Hours Studied | Hours Slept | Previous Score | Stress Level (1-10) | Pass? |
|---------------|-------------|----------------|---------------------|-------|
| 8 | 7 | 78 | 3 | 1 |
| 2 | 5 | 45 | 8 | 0 |
| 6 | 8 | 70 | 4 | 1 |
| 1 | 4 | 40 | 9 | 0 |
| 9 | 7 | 85 | 2 | 1 |
| 3 | 6 | 55 | 7 | 0 |
| 7 | 8 | 75 | 3 | 1 |
| 2 | 5 | 48 | 8 | 0 |

### Dataset Questions

**Q: A neural network with 1 hidden layer (4 neurons) is built for this dataset. How many trainable parameters are there?**
> **A:** Input layer: 4 features. Hidden layer: (4×4) weights + 4 biases = 20 params. Output layer: (4×1) weights + 1 bias = 5 params. **Total: 25 trainable parameters.** Formula: (n_inputs × n_hidden) + n_hidden + (n_hidden × n_output) + n_output.

**Q: After forward pass, the output neuron gives z=0.8 with sigmoid activation. What does the network predict?**
> **A:** σ(0.8) = 1/(1+e^(-0.8)) ≈ 0.69. Since 0.69 > 0.5 threshold, the network predicts **Pass = 1**.

**Q: The network predicts Pass=1 but actual is Pass=0. How does backpropagation fix this?**
> **A:** Backpropagation computes the gradient of the loss (cross-entropy) with respect to each weight. The gradient flows backward through the network, telling each weight whether to increase or decrease. Weights that contributed to the wrong prediction are adjusted via gradient descent to reduce future error.

---

## Questions & Answers

---

### Q1. What is Deep Learning?

**A:** Deep Learning is a subset of machine learning using neural networks with multiple layers (deep architectures) to automatically learn hierarchical representations from data. "Deep" refers to the many layers between input and output.

Key advantage: automatically learns features (no manual feature engineering needed for images, text, audio).

---

### Q2. What is an Artificial Neural Network (ANN)?

**A:** An ANN is a computational model loosely inspired by biological neural networks. It consists of:
- **Input layer:** Receives raw features
- **Hidden layers:** Transform features through learned weights + activations
- **Output layer:** Produces final prediction

Each neuron computes: `output = activation(Σ(weightᵢ × inputᵢ) + bias)`

---

### Q3. What are weights and biases?

**A:**
- **Weights (W):** Learnable parameters connecting neurons. A weight `wᵢⱼ` represents the strength of connection from neuron i to neuron j. Initialized randomly, updated by backpropagation.
- **Bias (b):** An additional learnable parameter added to the weighted sum. Allows the neuron to shift its activation independently of input. Like the intercept in linear regression.

`z = W·x + b` → activation → output

---

### Q4. What is backpropagation?

**A:** Backpropagation (backprop) is the algorithm for computing gradients of the loss with respect to all network parameters efficiently using the chain rule:

1. **Forward pass:** Compute predictions, calculate loss
2. **Backward pass:** From output to input, apply chain rule to propagate gradients
3. **Update:** Adjust weights using gradient descent: `W = W − α × ∂L/∂W`

Key insight: gradients flow backward through the network, each layer multiplies its local gradient with the incoming gradient.

---

### Q5. What is gradient descent?

**A:** Gradient descent minimizes the loss function by iteratively updating parameters in the direction of steepest descent:
```
θ = θ − α × ∇L(θ)
```
- `α` = learning rate
- `∇L(θ)` = gradient of loss w.r.t. parameters

**Variants:**
- **Batch GD:** Use all training data per update (stable, slow)
- **SGD:** Use 1 sample per update (fast, noisy)
- **Mini-batch GD:** Use small batches (best balance, default in deep learning)

---

### Q6. What are activation functions?

**A:** Activation functions introduce non-linearity into the network — without them, multiple layers collapse to a single linear transformation.

Without activation: `layer2(layer1(x)) = W2(W1x+b1)+b2 = (W2W1)x + c` (still linear)

Common activations: Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softmax

---

### Q7. What is the difference between Sigmoid, Tanh, and ReLU?

**A:**

| | Sigmoid | Tanh | ReLU |
|---|---|---|---|
| Formula | 1/(1+e^(-x)) | (e^x−e^(-x))/(e^x+e^(-x)) | max(0,x) |
| Output range | (0, 1) | (−1, 1) | [0, ∞) |
| Vanishing gradient | Yes (saturates at 0,1) | Yes (saturates at ±1) | No (positive region) |
| Zero-centered | No | Yes | No |
| Dying neuron | No | No | Yes (negative inputs) |
| Use case | Output (binary) | Hidden layers (older) | Hidden layers (default) |

ReLU is the default for hidden layers due to fast computation and no vanishing gradient in positive region.

---

### Q8. What is Batch Normalization?

**A:** Batch Normalization normalizes layer inputs within each mini-batch to have mean=0, std=1, then applies learnable scale (γ) and shift (β):

```
BN(x) = γ × (x − μ_batch) / √(σ²_batch + ε) + β
```

**Benefits:**
- Faster training (allows higher learning rates)
- Reduces internal covariate shift
- Acts as regularization (reduces need for dropout)
- Helps with vanishing/exploding gradients

---

### Q9. What is Dropout?

**A:** Dropout randomly deactivates (sets to zero) a fraction `p` of neurons during each training forward pass:

```python
nn.Dropout(p=0.5)  # 50% neurons dropped each step
```

**How it helps:**
- Forces network to learn redundant representations
- Acts as ensemble of 2^n sub-networks
- Prevents co-adaptation of neurons
- Reduces overfitting

At test time, all neurons are active and weights are scaled by (1−p).

---

### Q10. What is Transfer Learning?

**A:** Transfer Learning uses a model pretrained on one task as the starting point for a new related task:

1. Take pretrained model (e.g., ResNet trained on ImageNet)
2. Replace final layers for new task
3. Either:
   - **Feature extraction:** Freeze pretrained weights, only train new layers
   - **Fine-tuning:** Unfreeze and update some/all pretrained layers

Benefit: Much less data and compute needed. Pretrained features (edges, shapes, textures) transfer well.

---

### Q11. What is overfitting in neural networks?

**A:** Overfitting occurs when the model learns the training data too specifically (including noise) and fails to generalize. Signs:
- Training loss keeps decreasing
- Validation loss starts increasing (diverges from training loss)

**Prevention techniques:**
1. More training data
2. Dropout
3. L1/L2 weight regularization
4. Batch normalization
5. Early stopping
6. Data augmentation
7. Reduce model complexity

---

### Q12. What is the vanishing gradient problem?

**A:** During backpropagation, gradients are multiplied across layers. If gradients < 1 (e.g., sigmoid derivatives max at 0.25), they shrink exponentially as they flow backward:

```
Gradient at layer 1 = local_grad₁ × local_grad₂ × ... × local_gradₙ
If each ≈ 0.1: gradient = 0.1^10 = 10^-10 (essentially zero)
```

Deep layers barely update → network fails to learn long-range dependencies.

**Solutions:** ReLU activation, skip connections (ResNet), LSTM gates, gradient clipping, batch normalization.

---

### Q13. What is the exploding gradient problem?

**A:** The opposite: gradients grow exponentially during backpropagation, causing weights to update too aggressively → NaN values.

Common in:
- Very deep networks
- RNNs (long sequences)

**Solutions:**
- **Gradient clipping:** Clip gradient norm if it exceeds a threshold
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Careful weight initialization
- Batch normalization

---

### Q14. What is an optimizer?

**A:** An optimizer is the algorithm that updates model weights during training to minimize the loss. Beyond basic gradient descent, optimizers add techniques to:
- Adapt learning rates per parameter
- Accumulate momentum
- Escape local minima

Common optimizers: SGD, SGD+Momentum, RMSProp, Adam, AdamW, Lion

---

### Q15. What is the Adam optimizer?

**A:** Adam (Adaptive Moment Estimation) combines momentum and adaptive learning rates:

```
m_t = β₁ × m_{t-1} + (1−β₁) × g_t        # 1st moment (momentum)
v_t = β₂ × v_{t-1} + (1−β₂) × g_t²       # 2nd moment (adaptive LR)
m̂_t = m_t / (1−β₁ᵗ)                      # bias correction
v̂_t = v_t / (1−β₂ᵗ)                      # bias correction
θ = θ − α × m̂_t / (√v̂_t + ε)
```

Default params: `β₁=0.9, β₂=0.999, ε=1e-8, lr=0.001`

Best general-purpose optimizer for deep learning.

---

### Q16. What is a perceptron?

**A:** A perceptron is the simplest neural network — a single neuron:
```
output = step_function(w·x + b)
```
Can only learn **linearly separable** problems (famously fails on XOR). The limitation of single perceptrons led to multi-layer networks (MLPs).

---

### Q17. What is weight initialization and why does it matter?

**A:** Random initialization breaks symmetry (all-zero weights → all neurons learn the same). But:
- Too large: activations saturate, vanishing/exploding gradients
- Too small: signals vanish through layers

**Strategies:**
- **Xavier/Glorot:** `W ~ N(0, 1/n_in)` — good for sigmoid/tanh
- **He:** `W ~ N(0, 2/n_in)` — good for ReLU
- **Orthogonal:** For RNNs

```python
nn.Linear(4, 4)  # PyTorch uses Kaiming init by default for linear layers
torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

---

### Q18. What is the learning rate?

**A:** The learning rate α controls how large each gradient step is:
```
θ = θ − α × ∇L
```
- **Too high:** Overshoots minimum → diverges, loss oscillates
- **Too low:** Very slow convergence, may get stuck
- **Good value:** Requires tuning, typically 0.001–0.01 for Adam

**Learning rate schedule:** Decrease LR over time (step decay, cosine annealing, warmup).

---

### Q19. What is the Universal Approximation Theorem?

**A:** A neural network with one hidden layer with sufficient neurons can approximate any continuous function on a compact subset of ℝⁿ to arbitrary precision.

**Implication:** Deep networks are theoretically capable of learning any function — the question is whether they can learn it efficiently from finite data with optimization.

**Caveat:** "Sufficient neurons" can be exponentially large. Deeper networks are more practical (learn hierarchical features).

---

### Q20. What is the difference between regression and classification loss functions?

**A:**

**Classification:**
- Binary: `Binary Cross-Entropy = −[y log(ŷ) + (1−y) log(1−ŷ)]`
- Multi-class: `Categorical Cross-Entropy = −Σ yᵢ log(ŷᵢ)`

**Regression:**
- `MSE = (1/n) Σ(y − ŷ)²` — penalizes large errors heavily
- `MAE = (1/n) Σ|y − ŷ|` — robust to outliers
- `Huber = MSE if |error| ≤ δ else MAE` — best of both

---

## Quick Code Example

```python
import torch
import torch.nn as nn
import numpy as np

# Dataset
X = torch.tensor([[8,7,78,3],[2,5,45,8],[6,8,70,4],[1,4,40,9],
                  [9,7,85,2],[3,6,55,7],[7,8,75,3],[2,5,48,8]],
                  dtype=torch.float32)
y = torch.tensor([[1],[0],[1],[0],[1],[0],[1],[0]], dtype=torch.float32)

# Normalize
X = (X - X.mean(0)) / X.std(0)

# Define MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Training loop
for epoch in range(200):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        acc = ((preds > 0.5).float() == y).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")
```
