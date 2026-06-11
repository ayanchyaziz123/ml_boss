# Transformers & Attention — Interview Questions & Answers

---

## Small Dataset Example (Translation)

```
Source (English) → Target (French):

"I love cats"     → "J'aime les chats"
"She reads books" → "Elle lit des livres"
"We eat dinner"   → "Nous mangeons le dîner"
```

Token-level:
```
Input:  [I,    love,  cats]   → IDs: [10, 42, 7]
Output: [J',   aime,  les,  chats] → IDs: [5, 18, 3, 12]
```

### Dataset Questions

**Q: When the Transformer decoder generates "aime" (loves), which input token should it attend to most?**
> **A:** The decoder should attend most strongly to "love" in the encoder output. The cross-attention mechanism learns these alignment scores — high attention weight between "aime" and "love" because they're translations of each other.

**Q: Why does "cats" need to attend to itself and "love" in self-attention?**
> **A:** Self-attention allows each token to gather context from the entire sequence. "cats" is the object of "love" — attending to "love" helps understand the semantic role. Self-attention captures syntactic and semantic relationships that help better encode each word.

**Q: Without positional encoding, what problem would arise for "I love cats" vs "cats love I"?**
> **A:** Transformers process all tokens in parallel (no sequential order). Without positional encoding, both sequences produce identical representations — the model can't distinguish word order. Positional encoding adds position-specific signals so the model knows token 1 vs token 3.

---

## Questions & Answers

---

### Q1. What is the Transformer architecture?

**A:** The Transformer (Vaswani et al., 2017) is a neural network architecture based entirely on attention mechanisms, without recurrence or convolutions.

**Structure:**
- **Encoder:** N stacked layers, each with self-attention + feed-forward sublayers
- **Decoder:** N stacked layers, each with masked self-attention + cross-attention + feed-forward

Used in BERT (encoder), GPT (decoder), T5 (encoder-decoder).

---

### Q2. What is self-attention?

**A:** Self-attention allows each token to attend to all other tokens in the same sequence:

```
Q = X Wq  (queries)
K = X Wk  (keys)
V = X Wv  (values)

Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

Each token's output = weighted sum of all value vectors, where weights are similarity scores between its query and all keys.

Intuition: "How much should position i attend to position j?"

---

### Q3. What is multi-head attention?

**A:** Instead of one attention computation, multi-head attention runs h parallel attention heads:

```
head_i = Attention(Q Wq_i, K Wk_i, V Wv_i)
MultiHead(Q, K, V) = Concat(head₁,...,headₕ) Wₒ
```

Each head learns different types of relationships (syntactic, semantic, coreference, etc.). h=8 or h=12 heads are common.

---

### Q4. What is positional encoding?

**A:** Since Transformers process all tokens in parallel (no recurrence), they need an explicit signal for token position. Sinusoidal positional encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Added to token embeddings: `input = token_embed + position_embed`

Learned positional embeddings are also common (BERT uses learned positions).

---

### Q5. What is the encoder in a Transformer?

**A:** The encoder processes the entire input sequence in parallel. Each encoder layer has:

1. **Multi-head self-attention** — each token attends to all tokens
2. **Add & Norm** — residual connection + layer normalization
3. **Feed-forward network** — two linear layers with ReLU
4. **Add & Norm** — residual connection + layer normalization

Output: contextual representations for each input token (each token encodes its full context).

---

### Q6. What is the decoder in a Transformer?

**A:** The decoder generates output tokens autoregressively. Each decoder layer has:

1. **Masked self-attention** — attends to previous output tokens only (causal mask prevents future peeking)
2. **Add & Norm**
3. **Cross-attention** — attends to encoder output (aligns output with input)
4. **Add & Norm**
5. **Feed-forward network**
6. **Add & Norm**

---

### Q7. What is BERT?

**A:** BERT (Bidirectional Encoder Representations from Transformers) is a Transformer encoder pretrained with:
- **Masked Language Modeling (MLM):** Randomly mask 15% of tokens, predict them
- **Next Sentence Prediction (NSP):** Predict if two sentences follow each other

**Key:** Bidirectional — each token sees the full context (both left and right). Used for: text classification, NER, question answering.

```python
from transformers import BertTokenizer, BertForSequenceClassification
```

---

### Q8. What is GPT?

**A:** GPT (Generative Pretrained Transformer) is a Transformer decoder pretrained with **causal language modeling** — predict the next token given all previous tokens.

**Key:** Unidirectional (left to right). Good for text generation. GPT-2, GPT-3, GPT-4 scale this to billions of parameters.

```
Input:  "I love" → predict: "cats"
Training: P(wₜ | w₁,...,wₜ₋₁)
```

---

### Q9. What is the difference between BERT and GPT?

**A:**
| | BERT | GPT |
|---|---|---|
| Architecture | Encoder only | Decoder only |
| Context | Bidirectional (full context) | Unidirectional (left-to-right) |
| Pretraining | MLM + NSP | Causal LM (next token prediction) |
| Best for | Understanding tasks (classification, NER) | Generation tasks (text, code) |
| Attention mask | Full (no masking) | Causal (future masked) |

---

### Q10. What is masked language modeling?

**A:** MLM is BERT's pretraining objective:
1. Randomly mask 15% of input tokens with [MASK]
2. Train model to predict the original token

```
Input:  "The [MASK] sat on the mat"
Target: "The cat sat on the mat"
```

Forces the model to understand bidirectional context. The masking procedure: 80% → [MASK], 10% → random word, 10% → unchanged.

---

### Q11. What is scaled dot-product attention and why scale by √d_k?

**A:**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

**Why √d_k scaling?** The dot products QKᵀ grow in magnitude as d_k increases (variance of dot product = d_k for unit-variance inputs). Without scaling, large dot products push softmax into saturation → vanishing gradients. Dividing by √d_k normalizes the variance to 1.

---

### Q12. What is the feed-forward network in a Transformer?

**A:** Each Transformer layer has a position-wise feed-forward network applied identically to each position:

```
FFN(x) = max(0, x W₁ + b₁) W₂ + b₂
```

Typically: d_model=512, d_ff=2048 (4× expansion). Uses ReLU (original) or GELU (modern).

Purpose: Adds non-linearity and increases model capacity. The FFN actually holds most of the factual knowledge in language models.

---

### Q13. What is KV Cache?

**A:** During autoregressive generation, each new token requires computing attention over all previous tokens. KV Cache stores the Key and Value matrices for all previous tokens:

```
Without cache: Recompute K,V for all previous tokens at each step
With cache:    Store K,V, only compute for new token → O(1) per step
```

Speed improvement: ~2× for short contexts, much more for long contexts. Memory tradeoff: stores O(n × layers × 2 × d_model) values.

---

### Q14. What is the difference between encoder-only, decoder-only, and encoder-decoder?

**A:**
| Architecture | Models | Best For |
|---|---|---|
| Encoder-only | BERT, RoBERTa, DistilBERT | Classification, NER, QA extractive |
| Decoder-only | GPT-2/3/4, LLaMA, Mistral | Text generation, completion |
| Encoder-decoder | T5, BART, Marian | Translation, summarization, QA generative |

---

### Q15. What is fine-tuning a Transformer?

**A:** Take a pretrained Transformer and continue training on task-specific labeled data:

1. Load pretrained weights (e.g., `bert-base-uncased`)
2. Add task-specific head (classification layer)
3. Train on labeled data with small learning rate (e.g., 2e-5)
4. All weights update (full fine-tuning) or only top layers (frozen backbone)

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

---

### Q16. What is prompt engineering?

**A:** Prompt engineering is the practice of designing input text (prompts) to elicit desired outputs from a language model without changing its weights.

```
Zero-shot:  "Classify sentiment: 'I love this' →"
Few-shot:   "Positive: 'I love this'\nNegative: 'I hate this'\nLabel: 'This is great' →"
Chain-of-thought: "Let's think step by step..."
```

---

### Q17. What is zero-shot vs few-shot learning?

**A:**
- **Zero-shot:** Model performs task with only a task description, no examples
- **Few-shot:** Model gets 1–10 examples in the prompt (in-context learning)
- **Fine-tuning:** Model is trained on many labeled examples

LLMs (GPT-4, Claude) can do strong zero-shot and few-shot learning through in-context learning.

---

### Q18. What is RLHF?

**A:** Reinforcement Learning from Human Feedback trains language models to follow instructions and be helpful/harmless:

1. **Supervised Fine-Tuning (SFT):** Train on human-written demonstrations
2. **Reward Model (RM):** Train a reward model from human preference rankings
3. **PPO Optimization:** Use RL (PPO) to maximize reward model score

Used to train ChatGPT, Claude, Gemini. Dramatically improves instruction-following and reduces harmful outputs.

---

### Q19. What is LoRA (Low-Rank Adaptation)?

**A:** LoRA is a parameter-efficient fine-tuning method that adds low-rank update matrices:

```
W_new = W_pretrained + ΔW = W_pretrained + A × B
```
Where A ∈ ℝ^(d×r), B ∈ ℝ^(r×k) and rank r << min(d,k).

Only A and B are trained (much fewer parameters). W_pretrained is frozen.

With r=8: reduces trainable params by ~1000× compared to full fine-tuning. Enables fine-tuning 7B models on a single GPU.

---

### Q20. What is the context length and why does it matter?

**A:** Context length = maximum number of tokens the model can process in one forward pass.

- Standard BERT: 512 tokens
- GPT-3: 4096 tokens
- GPT-4: 128k tokens
- Claude 3: 200k tokens

**Bottleneck:** Self-attention is O(n²) in memory and compute — doubling context length quadruples cost. Extensions: sparse attention (Longformer), sliding window attention, RoPE for length generalization, Flash Attention for memory efficiency.

---

## Quick Code Example

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)

        out = (weights @ V).transpose(1, 2).contiguous()
        B, T, _, _ = out.shape
        out = out.view(B, T, -1)
        return self.W_o(out), weights

# Test
d_model, n_heads, seq_len, batch = 64, 4, 10, 2
mha = MultiHeadAttention(d_model, n_heads)
x = torch.randn(batch, seq_len, d_model)
out, attn_weights = mha(x, x, x)  # self-attention
print("Output shape:", out.shape)           # (2, 10, 64)
print("Attention weights shape:", attn_weights.shape)  # (2, 4, 10, 10)
```
