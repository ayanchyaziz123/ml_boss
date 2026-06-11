# Natural Language Processing (NLP) — Interview Questions & Answers

---

## Small Dataset Example

| Text | Label |
|------|-------|
| "The movie was absolutely fantastic!" | Positive |
| "Worst film I have ever seen" | Negative |
| "Pretty good, enjoyed watching it" | Positive |
| "Completely boring and predictable" | Negative |
| "Loved every scene, must watch" | Positive |
| "Terrible acting and bad script" | Negative |

Vocabulary: {the, movie, was, absolutely, fantastic, worst, film, ...}

### Dataset Questions

**Q: Apply TF-IDF to "fantastic". It appears in 1 out of 6 documents. What is the IDF?**
> **A:** IDF = log(N / df) = log(6/1) = log(6) ≈ **1.79**. Words rare across documents get higher IDF weight — "fantastic" is a strong signal. Common words like "the" appear in most documents → low IDF → down-weighted.

**Q: After tokenization and stop word removal of "The movie was absolutely fantastic!", what remains?**
> **A:** Remove stop words (the, was): **["movie", "absolutely", "fantastic"]**. Stop words like "the", "was", "is", "a" carry little meaning and are removed to focus on content words.

**Q: You want to represent "Loved every scene" as a vector for a classifier. Name three approaches from simplest to most powerful.**
> **A:** (1) **Bag of Words:** Count each word → sparse vector; (2) **TF-IDF:** Weighted count → sparse vector, focuses on important words; (3) **Sentence Embeddings (BERT/SBERT):** Dense 768-dim vector capturing full semantic meaning including context and word interactions.

---

## Questions & Answers

---

### Q1. What is NLP?

**A:** Natural Language Processing (NLP) is a branch of AI that enables computers to understand, process, and generate human language. Key tasks:
- Text classification, sentiment analysis
- Named entity recognition (NER)
- Machine translation
- Question answering
- Text summarization
- Information extraction

---

### Q2. What is tokenization?

**A:** Tokenization splits text into smaller units (tokens):

```
Word-level:    "I love NLP" → ["I", "love", "NLP"]
Subword (BPE): "unhappiness" → ["un", "happy", "ness"]
Character:     "cat" → ["c", "a", "t"]
```

Modern NLP uses **subword tokenization** (BPE, WordPiece, SentencePiece) — handles unknown words and rare words by breaking them into known subwords.

---

### Q3. What is stemming vs lemmatization?

**A:**

| | Stemming | Lemmatization |
|---|---|---|
| Method | Chop suffix heuristically | Look up dictionary form |
| Output | Stem (may not be valid word) | Lemma (valid word) |
| Speed | Faster | Slower |
| "running" | "run" | "run" |
| "better" | "better" | "good" |
| "studies" | "studi" | "study" |

Lemmatization is more accurate but slower. Stemming sufficient for search/retrieval.

---

### Q4. What is stop word removal?

**A:** Stop words are common words that carry little semantic meaning: "the", "is", "a", "an", "in", "on", "of". Removing them reduces vocabulary size and noise.

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [t for t in tokens if t not in stop_words]
```

**When NOT to remove:** Sentiment analysis ("not good" → removing "not" destroys meaning), machine translation, question answering.

---

### Q5. What is TF-IDF?

**A:** TF-IDF weights each word by its importance in a document relative to the corpus:

```
TF(t, d)  = count(t in d) / total_words(d)
IDF(t)    = log(N / df(t))   # N=docs, df=docs containing t
TF-IDF    = TF × IDF
```

- High TF: Word appears many times in document
- High IDF: Word is rare across all documents → discriminative
- Common words ("the"): Low IDF → low weight
- Rare, distinctive words: High IDF → high weight

---

### Q6. What is a Bag of Words (BoW)?

**A:** BoW represents a document as a vector of word counts, ignoring order:

```
Vocab: [movie, fantastic, boring, great, bad]
"great movie fantastic movie" → [2, 1, 0, 1, 0]
```

**Pros:** Simple, fast, works well for topic classification.
**Cons:** Ignores word order ("not good" ≠ "good not" treated same), sparse, no semantic similarity.

---

### Q7. What is a word embedding?

**A:** Word embeddings are dense, low-dimensional vector representations of words that capture semantic meaning:
- Similar words → nearby vectors
- Semantic relationships: king − man + woman ≈ queen

Trained on large corpora using self-supervised objectives (Word2Vec, GloVe). Dimension typically 50–300.

```python
# king - man + woman ≈ queen
model.wv['king'] - model.wv['man'] + model.wv['woman']
```

---

### Q8. What is Word2Vec?

**A:** Word2Vec (Mikolov et al., 2013) trains word embeddings using one of two objectives:

- **CBOW (Continuous Bag of Words):** Predict center word from context words
- **Skip-gram:** Predict context words from center word

```
CBOW:     [the, __, fast] → "very"
Skip-gram: "very" → [the, fast]
```

Skip-gram works better for rare words; CBOW is faster for frequent words. Training: negative sampling or hierarchical softmax.

---

### Q9. What is GloVe?

**A:** Global Vectors for Word Representation trains on the **co-occurrence matrix** of the corpus:

```
Objective: minimize ||wᵢᵀ w̃ⱼ + bᵢ + b̃ⱼ − log Xᵢⱼ||²
```

Where Xᵢⱼ = co-occurrence count of words i and j. 

Captures both local context (like Word2Vec) and global corpus statistics. Often outperforms Word2Vec on analogy tasks.

---

### Q10. What is Named Entity Recognition (NER)?

**A:** NER identifies and classifies named entities in text:

```
"Apple Inc. was founded by Steve Jobs in California."
→ [Apple Inc.: ORG, Steve Jobs: PERSON, California: LOCATION]
```

NER is sequence labeling — each token gets a label (B-ORG, I-ORG, B-PER, O, etc.).

Modern approach: fine-tuned BERT with token classification head.

---

### Q11. What is sentiment analysis?

**A:** Sentiment analysis classifies the opinion expressed in text:
- Binary: Positive / Negative
- Multi-class: Very Negative / Negative / Neutral / Positive / Very Positive
- Aspect-based: Sentiment toward specific aspects (food: positive, service: negative)

Approaches: Lexicon-based (VADER), ML (Logistic Regression + TF-IDF), Deep Learning (BERT fine-tuned).

---

### Q12. What is machine translation?

**A:** MT automatically translates text from source to target language. Evolution:
1. **Rule-based:** Hand-crafted linguistic rules
2. **Statistical MT:** Phrase-based statistical models
3. **Neural MT (NMT):** Encoder-decoder with attention
4. **Transformer-based:** State-of-the-art (Google Translate, DeepL)

**Key challenge:** Handling idioms, cultural context, ambiguity, and long-range dependencies.

---

### Q13. What is question answering (QA)?

**A:** QA systems answer questions from a context or knowledge base:
- **Extractive QA:** Find and return a span from the context (SQuAD task)
- **Abstractive QA:** Generate an answer (not necessarily verbatim)
- **Open-domain QA:** Retrieve relevant documents + extract answer

```
Context: "Paris is the capital of France..."
Question: "What is the capital of France?"
Answer: "Paris"  (extracted span)
```

BERT-based models dominate extractive QA.

---

### Q14. What is text summarization?

**A:**
- **Extractive summarization:** Select the most important sentences from the source text
- **Abstractive summarization:** Generate new text that captures the main points

Evaluation metrics: ROUGE (recall of n-grams), BLEU (precision of n-grams)

Models: BART, T5, PEGASUS (abstractive); TextRank, LSA (extractive).

---

### Q15. What is the BLEU score?

**A:** BLEU (Bilingual Evaluation Understudy) measures translation quality by comparing n-gram overlap between generated and reference translations:

```
BLEU = BP × exp(Σ wₙ × log pₙ)
```
- `pₙ` = n-gram precision
- `BP` = brevity penalty (penalize short translations)
- Range: 0 to 1 (higher = better)

BLEU=1.0: identical to reference. BLEU=0.3: decent human translation.
**Limitation:** Multiple correct translations penalized; doesn't measure fluency.

---

### Q16. What is the ROUGE score?

**A:** ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures summarization quality:

```
ROUGE-N Recall    = matching n-grams / total n-grams in reference
ROUGE-N Precision = matching n-grams / total n-grams in hypothesis
ROUGE-L           = longest common subsequence based
```

ROUGE focuses on recall (did the summary include key content?). BLEU focuses on precision.

---

### Q17. What is perplexity?

**A:** Perplexity measures how well a language model predicts a text sample — lower is better:

```
PPL(W) = P(w₁, w₂,...,wₙ)^(−1/n)
       = exp(−(1/n) Σ log P(wᵢ|w₁,...,wᵢ₋₁))
```

Interpretation: A perplexity of 50 means the model is as confused as if it had to choose uniformly among 50 words at each step. GPT-2 achieves ~29 on WikiText, GPT-4 much lower.

---

### Q18. What is BPE (Byte Pair Encoding) tokenization?

**A:** BPE is a subword tokenization algorithm:
1. Start with character vocabulary
2. Count most frequent adjacent pair
3. Merge that pair into a new token
4. Repeat until vocabulary size is reached

```
"low lower lowest" → chars: l, o, w, e, r, s, t
Merge "lo" → "lo"
Merge "low" → "low"
Final: ["low", "low##er", "low##est"]
```

Used by GPT-2, RoBERTa. Handles unknown words by breaking into known subwords.

---

### Q19. What is the difference between BERT and sentence embeddings (SBERT)?

**A:**
| | BERT (CLS token) | SBERT |
|---|---|---|
| Output | CLS token vector | Sentence-level embedding |
| Semantic similarity | Poor (CLS not trained for it) | Good (trained on NLI/paraphrase) |
| Comparison speed | O(n²) — must run model on each pair | O(1) — precompute and compare |
| Use case | Classification, NER | Semantic search, clustering |

SBERT uses siamese networks with contrastive loss to produce meaningful sentence embeddings.

---

### Q20. What is the difference between extractive and abstractive summarization?

**A:**

| | Extractive | Abstractive |
|---|---|---|
| Method | Select original sentences | Generate new sentences |
| Faithfulness | Always faithful (original text) | May hallucinate |
| Fluency | Depends on source quality | Usually more fluent |
| Complexity | Simpler (sentence ranking) | Harder (seq2seq) |
| Examples | TextRank, BertSum | BART, T5, PEGASUS |
| Best for | News articles, reports | Conversational, creative |

---

## Quick Code Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

texts = [
    "The movie was absolutely fantastic!",
    "Worst film I have ever seen",
    "Pretty good, enjoyed watching it",
    "Completely boring and predictable",
    "Loved every scene, must watch",
    "Terrible acting and bad script"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative

# TF-IDF Vectorization
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50, stop_words='english')
X = tfidf.fit_transform(texts)

print("Vocabulary size:", len(tfidf.vocabulary_))
print("Feature names:", tfidf.get_feature_names_out()[:10])

# Simple Logistic Regression classifier
clf = LogisticRegression()
clf.fit(X, labels)

preds = clf.predict(X)
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=['Negative', 'Positive']))

# HuggingFace pipeline (requires transformers)
# from transformers import pipeline
# classifier = pipeline("sentiment-analysis")
# result = classifier("This movie was absolutely amazing!")
# print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```
