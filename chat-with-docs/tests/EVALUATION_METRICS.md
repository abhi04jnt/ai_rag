# RAG Evaluation Metrics - Quick Reference

## 📊 Overview

This guide explains the evaluation metrics used in [test_evaluation.py](test_evaluation.py) to measure RAG system quality.

## 🧩 Chunk Quality Metrics

### 1. Semantic Coherence
**What**: Measures how semantically related consecutive sentences are within a chunk  
**How**: Average cosine similarity between consecutive sentence embeddings  
**Range**: 0.0 (unrelated) to 1.0 (identical)  
**Threshold**: >0.5 (good coherence)  
**Why**: Ensures chunks maintain topical consistency and don't split unrelated content together

### 2. Information Density
**What**: Ratio of unique words to total words  
**How**: `unique_words / total_words`  
**Range**: 0.0 (all repeated) to 1.0 (all unique)  
**Threshold**: >0.4 (adequate density)  
**Why**: Measures content richness - too low means repetitive text, too high might indicate poor chunking

### 3. Boundary Quality
**What**: Percentage of chunks that start with uppercase (not mid-sentence)  
**How**: Check if first character is uppercase  
**Range**: 0.0 (all bad) to 1.0 (all good)  
**Threshold**: >0.8 (80% clean boundaries)  
**Why**: Good boundaries preserve semantic units and improve retrieval quality

## 🔍 Retriever Metrics

### 1. Precision@K
**What**: Fraction of top-K retrieved documents that are relevant  
**Formula**: `relevant_in_top_k / k`  
**Range**: 0.0 (none relevant) to 1.0 (all relevant)  
**Threshold**: >0.4 for K=1 (40% accuracy at top result)  
**Why**: Measures retrieval accuracy - how many results are actually useful

**Example**:
- Retrieved top-5: [relevant, irrelevant, relevant, irrelevant, irrelevant]
- Precision@5 = 2/5 = 0.4

### 2. Recall@K
**What**: Fraction of all relevant documents found in top-K results  
**Formula**: `relevant_found / total_relevant`  
**Range**: 0.0 (none found) to 1.0 (all found)  
**Threshold**: >0.5 for K=10 (find 50%+ of relevant docs in top-10)  
**Why**: Measures retrieval completeness - did we find what we needed

**Example**:
- Total relevant docs: 3
- Found in top-10: 2
- Recall@10 = 2/3 = 0.67

### 3. Mean Reciprocal Rank (MRR)
**What**: Average of reciprocal ranks of first relevant document  
**Formula**: `average(1/rank_of_first_relevant)`  
**Range**: 0.0 (never found) to 1.0 (always rank 1)  
**Threshold**: >0.4 (first relevant doc typically in top 2-3)  
**Why**: Measures how quickly users find relevant results

**Example**:
- Query 1: First relevant at rank 1 → 1/1 = 1.0
- Query 2: First relevant at rank 3 → 1/3 = 0.33
- Query 3: First relevant at rank 2 → 1/2 = 0.5
- MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61

### 4. Normalized Discounted Cumulative Gain (NDCG@K)
**What**: Position-aware relevance metric that favors relevant docs at top  
**Formula**: `DCG / IDCG` where:
- DCG = Σ(relevance / log₂(rank + 1))
- IDCG = DCG with perfect ranking

**Range**: 0.0 (worst) to 1.0 (perfect ranking)  
**Threshold**: >0.5 for K=5  
**Why**: Better than precision/recall because it considers ranking quality

**Example**:
- Retrieved: [relevant, relevant, irrelevant, relevant, irrelevant]
- DCG = 1/log₂(2) + 1/log₂(3) + 0/log₂(4) + 1/log₂(5) + 0/log₂(6)
- DCG = 1.0 + 0.63 + 0 + 0.43 + 0 = 2.06
- IDCG = 1.0 + 0.63 + 0.5 = 2.13 (ideal: all relevant at top)
- NDCG = 2.06 / 2.13 = 0.97

## 💬 Generator Quality Metrics

### 1. Faithfulness
**What**: Degree to which answer is grounded in retrieved context  
**How**: Word overlap between answer and retrieved chunks  
**Formula**: `|answer_words ∩ context_words| / |answer_words|`  
**Range**: 0.0 (hallucinated) to 1.0 (fully grounded)  
**Threshold**: >0.5 (50%+ answer from context)  
**Why**: Prevents hallucination - ensures LLM uses provided information

### 2. Answer Relevance
**What**: Semantic similarity between question and answer  
**How**: Cosine similarity of question and answer embeddings  
**Range**: -1.0 to 1.0 (normalized embeddings: 0.0 to 1.0)  
**Threshold**: >0.3 (semantically related)  
**Why**: Ensures answer addresses the question, not off-topic

### 3. Key Concept Coverage
**What**: Percentage of required concepts present in answer  
**How**: Count how many must-have terms appear in answer  
**Formula**: `concepts_found / total_concepts`  
**Range**: 0.0 (none) to 1.0 (all)  
**Threshold**: >0.5 (covers 50%+ key points)  
**Why**: Ensures answer completeness - includes important information

## 🎯 End-to-End Metrics

### 1. Answer Correctness
**What**: Semantic similarity between generated answer and ground truth  
**How**: Cosine similarity of answer and ground truth embeddings  
**Range**: 0.0 (unrelated) to 1.0 (identical)  
**Threshold**: >0.6 (reasonably correct)  
**Why**: Overall quality measure - is the answer factually accurate?

### 2. RAG Latency
**What**: Time from question submission to answer return  
**Measured**: Average and P95 (95th percentile)  
**Threshold**: P95 < 10s  
**Why**: Ensures acceptable user experience

## 📈 Interpreting Results

### Good Performance
```
✅ Precision@1: 0.80 (80% of top results relevant)
✅ Recall@5: 0.90 (found 90% of relevant docs in top-5)
✅ MRR: 0.75 (first relevant typically at rank 1-2)
✅ NDCG@5: 0.85 (good ranking quality)
✅ Faithfulness: 0.70 (answers grounded in context)
✅ Answer Correctness: 0.75 (good semantic match to ground truth)
```

### Poor Performance
```
❌ Precision@1: 0.20 (only 20% top results relevant)
❌ Recall@5: 0.30 (missing 70% of relevant docs)
❌ MRR: 0.25 (first relevant at rank 4+)
❌ NDCG@5: 0.40 (poor ranking)
❌ Faithfulness: 0.30 (hallucinating, not using context)
❌ Answer Correctness: 0.40 (semantically distant from truth)
```

## 🛠️ Using the Metrics

### Run Evaluation
```bash
# All evaluation tests
pytest tests/test_evaluation.py -v -s

# Specific metric category
pytest tests/test_evaluation.py::TestRetrieverMetrics -v
pytest tests/test_evaluation.py::TestGeneratorQuality -v

# Single metric
pytest tests/test_evaluation.py::TestRetrieverMetrics::test_precision_at_k -v
```

### Interpreting Failures

**Low Precision/Recall**:
- Improve embeddings (use better model)
- Adjust chunking strategy (smaller/larger chunks)
- Try hybrid search (BM25 + vector)

**Low MRR/NDCG**:
- Fine-tune relevance scoring
- Improve re-ranking
- Adjust top_k parameter

**Low Faithfulness**:
- Reduce LLM temperature
- Improve system prompt (emphasize using context)
- Filter/verify retrieved chunks

**Low Answer Correctness**:
- Improve retrieval (fix Precision/Recall first)
- Better LLM model
- Enhance context formatting in prompt

## 📚 References

- **Precision/Recall**: Classic IR metrics from TREC evaluations
- **MRR**: Used by search engines (Google, Bing) for ranking quality
- **NDCG**: Standard metric for learning-to-rank systems
- **Faithfulness**: From RAG evaluation papers (e.g., FActScore, RAGAS)
- **Answer Correctness**: Semantic similarity widely used in QA evaluation

## 🎓 Further Reading

- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217)
- [BEIR: Benchmark for Information Retrieval](https://arxiv.org/abs/2104.08663)
- [FActScore: Fine-grained Atomic Evaluation](https://arxiv.org/abs/2305.14251)
- [TREC: Text Retrieval Conference](https://trec.nist.gov/)
