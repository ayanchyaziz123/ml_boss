# RNN, LSTM & GRU — Interview Questions & Answers

---

## Small Dataset Example (Sentiment Analysis)

| Sequence (words) | Sentiment |
|------------------|-----------|
| "I love this movie" | Positive |
| "Terrible waste of time" | Negative |
| "Amazing and wonderful film" | Positive |
| "Boring and slow plot" | Negative |
| "Highly recommend watching" | Positive |
| "Hated every minute" | Negative |

Encoded example:
```
Vocabulary: {I:1, love:2, this:3, movie:4, Terrible:5, waste:6, ...}
"I love this movie" → [1, 2, 3, 4] → Embeddings → RNN → Positive
```

### Dataset Questions

**Q: Why is an RNN better than an MLP for this task?**
> **A:** An MLP treats each word independently (bag-of-words). An RNN maintains a **hidden state** that carries context through the sequence — "not good" is understood differently than "good" because the hidden state remembers "not" when processing "good". Word order and sequential dependencies matter for sentiment.

**Q: "I love this movie" — how does the hidden state change as the RNN processes each word?**
> **A:** h₀ = zeros, h₁ = f(h₀, "I"), h₂ = f(h₁, "love"), h₃ = f(h₂, "this"), h₄ = f(h₃, "movie"). The final hidden state h₄ encodes the full sentence context and is used for the sentiment prediction.

**Q: For a sentence with 100 words, why might a vanilla RNN forget the first few words?**
> **A:** Vanilla RNNs suffer from **vanishing gradients** — gradients at word 1 pass through 99 multiplications during backpropagation. If each multiplication is < 1 (saturated activations), the gradient effectively disappears. The RNN "forgets" early words. LSTM solves this with a cell state and gating mechanism.

---

## Questions & Answers

---

### Q1. What is an RNN?

**A:** A Recurrent Neural Network (RNN) is a neural network that processes **sequential data** by maintaining a hidden state that gets updated at each time step:

```
hₜ = tanh(Wₕ × hₜ₋₁ + Wₓ × xₜ + b)
yₜ = Wᵧ × hₜ
```

The same weights (Wₕ, Wₓ) are shared across all time steps — this is what makes it recurrent. The hidden state `hₜ` serves as the memory of the sequence.

---

### Q2. What is a hidden state?

**A:** The hidden state `hₜ` is the RNN's memory — a vector that summarizes all the information from the sequence up to time step t. It's passed from one time step to the next:

```
h₀ → h₁ → h₂ → ... → hₜ (final context vector)
```

For classification, the final hidden state is typically used as the sequence representation.

---

### Q3. What are the limitations of vanilla RNN?

**A:**
1. **Vanishing gradient:** Gradients shrink exponentially through long sequences → can't learn long-range dependencies
2. **Exploding gradient:** Gradients grow exponentially (fixed with gradient clipping)
3. **Short-term memory:** Forgets information from early sequence steps
4. **Slow training:** Sequential computation — can't parallelize within a sequence
5. **Unstable training:** Sensitive to initialization

---

### Q4. What is LSTM?

**A:** Long Short-Term Memory (LSTM) solves the vanishing gradient problem with a **cell state** (long-term memory) and three gates that control information flow:

- **Forget gate (f):** What to erase from cell state
- **Input gate (i):** What new info to add to cell state
- **Output gate (o):** What to output to hidden state

```
fₜ = σ(Wf[hₜ₋₁, xₜ] + bf)      # forget gate
iₜ = σ(Wi[hₜ₋₁, xₜ] + bi)      # input gate
C̃ₜ = tanh(Wc[hₜ₋₁, xₜ] + bc)  # candidate cell
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ     # cell state update
oₜ = σ(Wo[hₜ₋₁, xₜ] + bo)      # output gate
hₜ = oₜ ⊙ tanh(Cₜ)             # hidden state
```

---

### Q5. What is the forget gate?

**A:** The forget gate decides what information to **remove** from the cell state. It outputs values between 0 and 1 using sigmoid:
- f=1: Keep this information completely
- f=0: Erase this information completely
- f=0.5: Keep half

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
Cₜ_new = fₜ ⊙ Cₜ₋₁  ← selectively forgetting
```

Example: In a language model, when it sees a period, the forget gate clears subject/verb tense information since a new sentence begins.

---

### Q6. What is the input gate?

**A:** The input gate decides what **new information** to add to the cell state — composed of two parts:
1. **Input gate (iₜ):** Which values to update (0=ignore, 1=update)
2. **Candidate values (C̃ₜ):** What the new content would be

```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)   # what to update
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  # candidate content
Cell update: Cₜ += iₜ ⊙ C̃ₜ
```

---

### Q7. What is the output gate?

**A:** The output gate controls what part of the cell state to expose as the hidden state:

```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ ⊙ tanh(Cₜ)
```

The cell state may hold many things (long-term memory); the output gate selects the relevant part for the current output/decision.

---

### Q8. What is the cell state in LSTM?

**A:** The cell state Cₜ is the LSTM's **long-term memory** — a highway that runs across all time steps with only linear interactions (forget gate × old state + input gate × new content). No squashing with tanh → gradients can flow back many steps without vanishing. This is the key innovation that allows LSTMs to learn long-range dependencies.

---

### Q9. What is GRU?

**A:** Gated Recurrent Unit (GRU) is a simplified LSTM with only two gates and no separate cell state:

```
zₜ = σ(Wz · [hₜ₋₁, xₜ])     # update gate (combines forget+input)
rₜ = σ(Wr · [hₜ₋₁, xₜ])     # reset gate
h̃ₜ = tanh(W · [rₜ⊙hₜ₋₁, xₜ]) # candidate hidden state
hₜ = (1−zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ  # hidden state
```

- **Update gate (z):** How much of past hidden state to keep vs. replace
- **Reset gate (r):** How much of past state to forget when computing candidate

---

### Q10. What is the difference between LSTM and GRU?

**A:**
| | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Memory | Separate cell state + hidden state | Single hidden state |
| Parameters | More (4× weight matrices) | Fewer (3× weight matrices) |
| Performance | Slightly better on complex tasks | Often comparable |
| Training speed | Slower | Faster (fewer parameters) |
| Memory | More | Less |

**When to use:** GRU for smaller datasets and faster training; LSTM for complex long-range dependencies.

---

### Q11. What is a bidirectional RNN?

**A:** A bidirectional RNN runs two RNNs — one forward (left to right) and one backward (right to left) — and concatenates their hidden states:

```
h_forward:   x₁ → x₂ → x₃ → x₄
h_backward:  x₄ → x₃ → x₂ → x₁
hₜ = [h_forward_t ; h_backward_t]
```

Benefit: Each position has context from both past and future — useful for NLP tasks where future context helps (NER, POS tagging, BERT uses this idea).

```python
nn.LSTM(input_size, hidden_size, bidirectional=True)
# Output size: hidden_size × 2
```

---

### Q12. What is sequence-to-sequence (seq2seq) model?

**A:** A seq2seq model maps an input sequence to an output sequence (potentially different length):
- **Encoder:** LSTM reads input → produces context vector (final hidden state)
- **Decoder:** LSTM generates output one token at a time, conditioned on context vector

Applications: Machine translation, text summarization, chatbots.

**Bottleneck:** The entire input is compressed into a single fixed-length context vector — information loss for long sequences. Solved by **attention mechanism**.

---

### Q13. What is teacher forcing?

**A:** During training, instead of feeding the decoder's own predictions as the next input, feed the **ground truth** labels:

- Without teacher forcing: decoder error compounds at each step
- With teacher forcing: stable training, faster convergence

**Drawback:** At inference time, the model sees its own (possibly wrong) outputs → train-test mismatch (exposure bias). Fix: **scheduled sampling** (gradually reduce teacher forcing ratio during training).

---

### Q14. What is the vanishing gradient problem in RNNs?

**A:** During BPTT (backpropagation through time), gradients are multiplied by the weight matrix at each time step. If the largest singular value of W < 1:

```
gradient at step 1 = W^T × ... × gradient at step T
```

Over T steps: gradient approaches 0 exponentially. The RNN cannot learn that something at step 1 influences the output at step T.

**Fix:** LSTM/GRU (gated memory), gradient clipping, transformer architectures.

---

### Q15. What is Backpropagation Through Time (BPTT)?

**A:** BPTT is the backpropagation algorithm applied to unrolled RNNs. The RNN is unrolled across T time steps, becoming a deep feedforward network, and standard backprop is applied.

**Truncated BPTT:** Instead of backpropagating through all T steps, backpropagate only through the last k steps. Reduces memory and computation — practical for very long sequences.

---

### Q16. What is stacked LSTM?

**A:** Multiple LSTM layers stacked on top of each other, where the output of one LSTM is the input to the next:

```python
nn.LSTM(input_size=64, hidden_size=128, num_layers=3, batch_first=True)
```

Layer 1 learns low-level patterns; Layer 2 learns patterns of patterns; etc. Generally improves performance but adds parameters and risk of overfitting.

---

### Q17. What is attention mechanism in RNN context?

**A:** Attention allows the decoder to look at **all encoder hidden states** (not just the final one) and learn which to focus on for each output step:

```
score(hₜ, h̄ₛ) = hₜᵀWh̄ₛ  (learned alignment score)
αₜₛ = softmax(scores)     (attention weights)
context_t = Σₛ αₜₛ × h̄ₛ  (weighted sum of encoder states)
```

This allows the decoder to "attend" to relevant input positions — solved the seq2seq bottleneck and inspired Transformers.

---

### Q18. What is the difference between LSTM and Transformer?

**A:**
| | LSTM | Transformer |
|---|---|---|
| Parallelism | Sequential (can't parallelize within sequence) | Fully parallel |
| Long-range dependencies | Limited (still degrades) | Full (all positions attend to all) |
| Training speed | Slow | Fast (GPU-parallelized) |
| Memory | O(sequence length) | O(sequence length²) attention matrix |
| State-of-the-art NLP | Replaced by | Current standard |
| Streaming/real-time | Better (sequential) | Harder |

---

### Q19. What is gradient clipping?

**A:** Gradient clipping prevents exploding gradients by scaling down the gradient when its norm exceeds a threshold:

```python
# Clip by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

Essential for RNNs. For LSTM/GRU: clip by norm with max_norm=1.0–5.0 is standard practice.

---

### Q20. What is a language model and how is it built with LSTM?

**A:** A language model predicts the next token given previous tokens: P(wₜ | w₁, w₂, ..., wₜ₋₁)

LSTM language model:
1. Embed each word → dense vector
2. LSTM processes sequence → hidden state at each position
3. Linear layer + softmax over vocabulary → probability of next word
4. Train with cross-entropy loss (predicting next word)

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc    = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)  # logits over vocab at each position
```

---

## Quick Code Example

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, n_layers,
                              batch_first=True, dropout=0.3,
                              bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc    = nn.Linear(hidden_size * 2, 1)  # ×2 for bidirectional

    def forward(self, x):
        embedded = self.dropout(self.embed(x))
        out, (hidden, _) = self.lstm(embedded)
        # Concatenate last forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return torch.sigmoid(self.fc(self.dropout(hidden)))

# Toy example
vocab_size = 100
model = SentimentLSTM(vocab_size)

# Fake batch: 4 sequences of length 10
x = torch.randint(1, vocab_size, (4, 10))
preds = model(x)
print("Predictions shape:", preds.shape)  # (4, 1)
print("Sample predictions:", preds.detach().flatten())

# GRU version (simpler)
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru   = nn.GRU(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc    = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        _, hidden = self.gru(self.embed(x))
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return torch.sigmoid(self.fc(hidden))
```
