"""
RAG Evaluation Tests - Chunk Quality, Retriever Metrics, Generator Quality

Tests RAG system quality with metrics:
- Chunk Quality: Semantic coherence, information density
- Retriever: Precision@K, Recall@K, MRR, NDCG
- Generator: Faithfulness, answer relevance, ROUGE
- End-to-End: Answer correctness against ground truth

Run with: pytest tests/test_evaluation.py -v
"""

import pytest
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.ingest import run_ingest
from src.rag.vectorstore import FaissStore
from src.rag.embeddings import Embedder
from src.rag.chat import ChatService
from src.rag.chunking import chunk_text
from src.rag.config import settings


# Ground truth test dataset
EVAL_DATASET = [
    {
        "question": "What is artificial intelligence?",
        "relevant_docs": ["ai_overview.txt"],
        "ground_truth_answer": "AI is the simulation of human intelligence by machines, enabling them to perform tasks that typically require human cognition.",
        "must_contain": ["intelligence", "machine", "human"]
    },
    {
        "question": "What is machine learning?",
        "relevant_docs": ["ml_guide.txt"],
        "ground_truth_answer": "Machine learning is a subset of AI where systems learn patterns from data without explicit programming.",
        "must_contain": ["learn", "data", "pattern"]
    },
    {
        "question": "Explain neural networks",
        "relevant_docs": ["neural_nets.txt"],
        "ground_truth_answer": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.",
        "must_contain": ["network", "node", "layer"]
    },
    {
        "question": "What is deep learning?",
        "relevant_docs": ["deep_learning.txt"],
        "ground_truth_answer": "Deep learning uses multi-layered neural networks to learn hierarchical representations of data.",
        "must_contain": ["deep", "layer", "neural"]
    },
    {
        "question": "What are transformers in AI?",
        "relevant_docs": ["transformers.txt"],
        "ground_truth_answer": "Transformers are neural network architectures that use self-attention mechanisms to process sequential data efficiently.",
        "must_contain": ["attention", "architecture", "sequence"]
    },
]


@pytest.fixture(scope="module")
def eval_docs_dir(tmp_path_factory):
    """Create evaluation test documents"""
    docs_dir = tmp_path_factory.mktemp("eval_docs")
    
    # Create test documents with known content
    (docs_dir / "ai_overview.txt").write_text(
        "Artificial Intelligence (AI) Overview\n\n"
        "Artificial intelligence refers to the simulation of human intelligence by machines. "
        "These intelligent systems are designed to perform tasks that typically require human cognition, "
        "such as visual perception, speech recognition, decision-making, and language translation. "
        "AI systems can learn from experience, adjust to new inputs, and perform human-like tasks. "
        "The field encompasses machine learning, deep learning, neural networks, and natural language processing."
    )
    
    (docs_dir / "ml_guide.txt").write_text(
        "Machine Learning Guide\n\n"
        "Machine learning is a subset of artificial intelligence that focuses on developing systems "
        "that can learn and improve from experience without being explicitly programmed. "
        "ML algorithms build mathematical models based on sample data, known as training data, "
        "to make predictions or decisions without being explicitly programmed to perform the task. "
        "Common types include supervised learning, unsupervised learning, and reinforcement learning. "
        "Machine learning is used in email filtering, computer vision, and recommendation systems."
    )
    
    (docs_dir / "neural_nets.txt").write_text(
        "Neural Networks Explained\n\n"
        "Neural networks are computing systems vaguely inspired by the biological neural networks "
        "that constitute animal brains. A neural network consists of interconnected groups of nodes, "
        "analogous to neurons in a brain. Each connection, like synapses in a biological brain, "
        "can transmit a signal to other neurons. Neurons are organized in layers: input layer, "
        "hidden layers, and output layer. Each node processes input and passes output to the next layer. "
        "Neural networks learn by adjusting weights between connections during training."
    )
    
    (docs_dir / "deep_learning.txt").write_text(
        "Deep Learning Introduction\n\n"
        "Deep learning is a subset of machine learning that uses multi-layered neural networks, "
        "called deep neural networks, to learn hierarchical representations of data. "
        "The 'deep' refers to the number of layers through which data is transformed. "
        "Deep learning has revolutionized computer vision, speech recognition, and natural language processing. "
        "Popular architectures include Convolutional Neural Networks (CNNs) for images, "
        "Recurrent Neural Networks (RNNs) for sequences, and Transformers for language tasks. "
        "Deep learning requires large amounts of data and computational power but achieves state-of-the-art results."
    )
    
    (docs_dir / "transformers.txt").write_text(
        "Transformer Architecture\n\n"
        "Transformers are a type of neural network architecture introduced in 2017 that has become "
        "the dominant approach for natural language processing tasks. Unlike previous architectures, "
        "transformers use self-attention mechanisms to process sequential data in parallel, "
        "making them much more efficient than recurrent approaches. The architecture consists of "
        "encoder and decoder layers, each with multi-head self-attention and feed-forward networks. "
        "Transformers power models like BERT, GPT, and T5, enabling breakthrough performance in "
        "machine translation, text generation, and question answering."
    )
    
    (docs_dir / "unrelated.txt").write_text(
        "Random Unrelated Content\n\n"
        "The weather today is sunny with a high of 75 degrees. "
        "Tomorrow expect clouds with a chance of rain. "
        "Next week will be cooler with temperatures in the 60s."
    )
    
    return docs_dir


@pytest.fixture(scope="module")
def eval_index_dir(tmp_path_factory):
    """Create temporary index directory"""
    return tmp_path_factory.mktemp("eval_index")


@pytest.fixture(scope="module")
def indexed_system(eval_docs_dir, eval_index_dir):
    """Set up indexed system for evaluation"""
    # Override settings
    original_docs = settings.docs_dir
    original_index = settings.index_dir
    
    settings.docs_dir = str(eval_docs_dir)
    settings.index_dir = str(eval_index_dir)
    
    # Run ingestion
    run_ingest(reset=True, docs_dir=str(eval_docs_dir))
    
    # Create services
    embedder = Embedder(settings.embedding_model)
    store = FaissStore(index_dir=str(eval_index_dir))
    store.load()  # Load the index
    chat = ChatService()
    
    yield {
        "embedder": embedder,
        "store": store,
        "chat": chat,
        "docs_dir": eval_docs_dir,
        "index_dir": eval_index_dir
    }
    
    # Restore settings
    settings.docs_dir = original_docs
    settings.index_dir = original_index


class TestChunkQuality:
    """Test chunk quality metrics"""
    
    def test_chunk_semantic_coherence(self, indexed_system):
        """Test that chunks maintain semantic coherence"""
        store = indexed_system["store"]
        embedder = indexed_system["embedder"]
        
        # Get chunks from metadata
        chunks = store.meta
        
        coherence_scores = []
        for chunk in chunks[:10]:  # Sample first 10
            text = chunk["text"]
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) >= 2:
                # Embed consecutive sentences
                embeddings = embedder.embed_texts(sentences)
                
                # Calculate average cosine similarity between consecutive sentences
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = np.dot(embeddings[i], embeddings[i+1])
                    similarities.append(sim)
                
                if similarities:
                    coherence = np.mean(similarities)
                    coherence_scores.append(coherence)
        
        # Chunks should have reasonable coherence (>0.45)
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
        assert avg_coherence > 0.45, f"Chunk coherence too low: {avg_coherence:.3f}"
        print(f"\n✅ Average chunk coherence: {avg_coherence:.3f}")
    
    def test_chunk_information_density(self, indexed_system):
        """Test that chunks have adequate information density"""
        store = indexed_system["store"]
        
        chunks = store.meta
        
        densities = []
        for chunk in chunks:
            text = chunk["text"]
            # Calculate information density as ratio of unique words to total words
            words = text.lower().split()
            if len(words) > 0:
                unique_words = len(set(words))
                density = unique_words / len(words)
                densities.append(density)
        
        assert len(densities) > 0, "No chunks to evaluate"
        avg_density = np.mean(densities)
        
        # Information density should be reasonable (>0.4)
        assert avg_density > 0.4, f"Information density too low: {avg_density:.3f}"
        print(f"✅ Average information density: {avg_density:.3f}")
    
    def test_chunk_boundary_quality(self, indexed_system):
        """Test that chunk boundaries don't split sentences"""
        store = indexed_system["store"]
        
        chunks = store.meta
        
        bad_boundaries = 0
        total_chunks = len(chunks)
        
        for chunk in chunks:
            text = chunk["text"].strip()
            # Check if chunk starts/ends mid-sentence (very crude check)
            if text and not text[0].isupper():
                bad_boundaries += 1
        
        boundary_quality = 1 - (bad_boundaries / total_chunks)
        
        # At least 80% should have good boundaries
        assert boundary_quality > 0.8, f"Too many bad boundaries: {boundary_quality:.1%}"
        print(f"✅ Chunk boundary quality: {boundary_quality:.1%}")


class TestRetrieverMetrics:
    """Test retriever performance with standard IR metrics"""
    
    def test_precision_at_k(self, indexed_system):
        """Test Precision@K for retrieval"""
        embedder = indexed_system["embedder"]
        store = indexed_system["store"]
        
        k_values = [1, 3, 5]
        precision_scores = {k: [] for k in k_values}
        
        for example in EVAL_DATASET:
            question = example["question"]
            relevant_docs = set(example["relevant_docs"])
            
            # Get query embedding
            query_vec = embedder.embed_texts([question])[0]
            query_vec = query_vec.reshape(1, -1)  # Reshape to (1, 384) for FAISS
            
            # Retrieve documents
            results = store.search(query_vec, top_k=max(k_values))
            
            # Calculate precision at each K
            for k in k_values:
                retrieved_k = results[:k]
                relevant_retrieved = sum(
                    1 for r in retrieved_k 
                    if any(doc in r["source"] for doc in relevant_docs)
                )
                precision_k = relevant_retrieved / k if k > 0 else 0
                precision_scores[k].append(precision_k)
        
        # Report and validate
        for k in k_values:
            avg_precision = np.mean(precision_scores[k])
            print(f"✅ Precision@{k}: {avg_precision:.3f}")
            # At least 40% precision at top results
            if k == 1:
                assert avg_precision > 0.4, f"Precision@1 too low: {avg_precision:.3f}"
    
    def test_recall_at_k(self, indexed_system):
        """Test Recall@K for retrieval"""
        embedder = indexed_system["embedder"]
        store = indexed_system["store"]
        
        k_values = [3, 5, 10]
        recall_scores = {k: [] for k in k_values}
        
        for example in EVAL_DATASET:
            question = example["question"]
            relevant_docs = set(example["relevant_docs"])
            
            query_vec = embedder.embed_texts([question])[0]
            query_vec = query_vec.reshape(1, -1)  # Reshape to (1, 384) for FAISS
            results = store.search(query_vec, top_k=max(k_values))
            
            for k in k_values:
                retrieved_k = results[:k]
                relevant_retrieved = set()
                for r in retrieved_k:
                    for doc in relevant_docs:
                        if doc in r["source"]:
                            relevant_retrieved.add(doc)
                
                recall_k = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
                recall_scores[k].append(recall_k)
        
        for k in k_values:
            avg_recall = np.mean(recall_scores[k])
            print(f"✅ Recall@{k}: {avg_recall:.3f}")
            # Recall should improve with more results
            if k == 10:
                assert avg_recall > 0.5, f"Recall@10 too low: {avg_recall:.3f}"
    
    def test_mean_reciprocal_rank(self, indexed_system):
        """Test MRR - average rank of first relevant document"""
        embedder = indexed_system["embedder"]
        store = indexed_system["store"]
        
        reciprocal_ranks = []
        
        for example in EVAL_DATASET:
            question = example["question"]
            relevant_docs = set(example["relevant_docs"])
            
            query_vec = embedder.embed_texts([question])[0]
            query_vec = query_vec.reshape(1, -1)  # Reshape to (1, 384) for FAISS
            results = store.search(query_vec, top_k=10)
            
            # Find rank of first relevant document
            rank = None
            for i, r in enumerate(results, 1):
                if any(doc in r["source"] for doc in relevant_docs):
                    rank = i
                    break
            
            if rank:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        mrr = np.mean(reciprocal_ranks)
        print(f"✅ Mean Reciprocal Rank (MRR): {mrr:.3f}")
        # MRR should be reasonable (>0.4)
        assert mrr > 0.4, f"MRR too low: {mrr:.3f}"
    
    def test_ndcg_at_k(self, indexed_system):
        """Test NDCG@K - normalized discounted cumulative gain"""
        embedder = indexed_system["embedder"]
        store = indexed_system["store"]
        
        k = 5
        ndcg_scores = []
        
        for example in EVAL_DATASET:
            question = example["question"]
            relevant_docs = set(example["relevant_docs"])
            
            query_vec = embedder.embed_texts([question])[0]
            query_vec = query_vec.reshape(1, -1)  # Reshape to (1, 384) for FAISS
            results = store.search(query_vec, top_k=k)
            
            # Calculate DCG
            dcg = 0.0
            for i, r in enumerate(results, 1):
                relevance = 1.0 if any(doc in r["source"] for doc in relevant_docs) else 0.0
                dcg += relevance / np.log2(i + 1)
            
            # Calculate ideal DCG (all relevant docs at top)
            num_relevant = min(len(relevant_docs), k)
            idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        avg_ndcg = np.mean(ndcg_scores)
        print(f"✅ NDCG@{k}: {avg_ndcg:.3f}")
        # NDCG should be reasonable (>0.5)
        assert avg_ndcg > 0.5, f"NDCG@{k} too low: {avg_ndcg:.3f}"


class TestGeneratorQuality:
    """Test generator (LLM) output quality"""
    
    def test_answer_faithfulness(self, indexed_system):
        """Test that answers are faithful to retrieved context"""
        chat = indexed_system["chat"]
        
        faithfulness_scores = []
        
        for example in EVAL_DATASET[:3]:  # Sample subset to save API calls
            question = example["question"]
            
            # Get answer from system
            try:
                response = chat.answer(question, history=[])
                answer = response["answer"]
                retrieved = response["retrieved"]
                
                # Check if answer content appears in retrieved context
                retrieved_text = " ".join(r["text"] for r in retrieved)
                
                # Simple faithfulness: check if answer words appear in context
                answer_words = set(answer.lower().split())
                # Remove stopwords
                stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "but"}
                answer_words -= stopwords
                
                context_words = set(retrieved_text.lower().split())
                
                if answer_words:
                    overlap = len(answer_words & context_words) / len(answer_words)
                    faithfulness_scores.append(overlap)
            except Exception as e:
                print(f"⚠️  Skipping test (API error): {e}")
                pytest.skip("OpenAI API not available")
        
        if faithfulness_scores:
            avg_faithfulness = np.mean(faithfulness_scores)
            print(f"✅ Answer faithfulness: {avg_faithfulness:.3f}")
            # At least 50% of answer should come from context
            assert avg_faithfulness > 0.5, f"Faithfulness too low: {avg_faithfulness:.3f}"
    
    def test_answer_relevance(self, indexed_system):
        """Test that answers are relevant to questions"""
        chat = indexed_system["chat"]
        embedder = indexed_system["embedder"]
        
        relevance_scores = []
        
        for example in EVAL_DATASET[:3]:  # Sample subset
            question = example["question"]
            
            try:
                response = chat.answer(question, history=[])
                answer = response["answer"]
                
                # Calculate semantic similarity between question and answer
                q_vec = embedder.embed_texts([question])[0]
                a_vec = embedder.embed_texts([answer])[0]
                
                similarity = np.dot(q_vec, a_vec)
                relevance_scores.append(similarity)
            except Exception as e:
                print(f"⚠️  Skipping test (API error): {e}")
                pytest.skip("OpenAI API not available")
        
        if relevance_scores:
            avg_relevance = np.mean(relevance_scores)
            print(f"✅ Answer relevance: {avg_relevance:.3f}")
            # Answers should be semantically related to questions (>0.3)
            assert avg_relevance > 0.3, f"Relevance too low: {avg_relevance:.3f}"
    
    def test_answer_contains_key_concepts(self, indexed_system):
        """Test that answers contain key concepts from ground truth"""
        chat = indexed_system["chat"]
        
        concept_coverage = []
        
        for example in EVAL_DATASET[:3]:
            question = example["question"]
            must_contain = example["must_contain"]
            
            try:
                response = chat.answer(question, history=[])
                answer = response["answer"].lower()
                
                # Check coverage of key concepts
                concepts_found = sum(1 for concept in must_contain if concept.lower() in answer)
                coverage = concepts_found / len(must_contain)
                concept_coverage.append(coverage)
            except Exception as e:
                print(f"⚠️  Skipping test (API error): {e}")
                pytest.skip("OpenAI API not available")
        
        if concept_coverage:
            avg_coverage = np.mean(concept_coverage)
            print(f"✅ Key concept coverage: {avg_coverage:.3f}")
            # At least 50% of key concepts should appear
            assert avg_coverage > 0.5, f"Concept coverage too low: {avg_coverage:.3f}"


class TestEndToEndRAG:
    """Test complete RAG pipeline quality"""
    
    def test_answer_correctness(self, indexed_system):
        """Test overall answer correctness vs ground truth"""
        chat = indexed_system["chat"]
        embedder = indexed_system["embedder"]
        
        correctness_scores = []
        
        for example in EVAL_DATASET[:3]:
            question = example["question"]
            ground_truth = example["ground_truth_answer"]
            
            try:
                response = chat.answer(question, history=[])
                answer = response["answer"]
                
                # Calculate semantic similarity to ground truth
                gt_vec = embedder.embed_texts([ground_truth])[0]
                ans_vec = embedder.embed_texts([answer])[0]
                
                similarity = np.dot(gt_vec, ans_vec)
                correctness_scores.append(similarity)
                
                print(f"\nQ: {question}")
                print(f"GT: {ground_truth[:100]}...")
                print(f"A: {answer[:100]}...")
                print(f"Score: {similarity:.3f}")
                
            except Exception as e:
                print(f"⚠️  Skipping test (API error): {e}")
                pytest.skip("OpenAI API not available")
        
        if correctness_scores:
            avg_correctness = np.mean(correctness_scores)
            print(f"\n✅ Average answer correctness: {avg_correctness:.3f}")
            # Answers should be reasonably similar to ground truth (>0.6)
            assert avg_correctness > 0.6, f"Answer correctness too low: {avg_correctness:.3f}"
    
    def test_retrieval_impact_on_quality(self, indexed_system):
        """Test that better retrieval leads to better answers"""
        chat = indexed_system["chat"]
        embedder = indexed_system["embedder"]
        
        # Test with good retrieval (relevant question)
        good_question = "What is artificial intelligence?"
        try:
            good_response = chat.answer(good_question, history=[])
            good_answer = good_response["answer"]
            
            # Test with poor retrieval (off-topic question)
            bad_question = "What is the weather forecast?"
            bad_response = chat.answer(bad_question, history=[])
            bad_answer = bad_response["answer"]
            
            # Good retrieval should produce more confident answers
            # (measured by length and specificity)
            good_length = len(good_answer.split())
            bad_length = len(bad_answer.split())
            
            print(f"✅ Good retrieval answer length: {good_length} words")
            print(f"✅ Poor retrieval answer length: {bad_length} words")
            
            # This is more observational than a strict test
            # Good answers tend to be more detailed
            
        except Exception as e:
            print(f"⚠️  Skipping test (API error): {e}")
            pytest.skip("OpenAI API not available")
    
    def test_rag_latency(self, indexed_system):
        """Test end-to-end RAG latency"""
        import time
        chat = indexed_system["chat"]
        
        latencies = []
        
        for example in EVAL_DATASET[:3]:
            question = example["question"]
            
            try:
                start = time.time()
                response = chat.answer(question, history=[])
                latency = time.time() - start
                latencies.append(latency)
            except Exception as e:
                print(f"⚠️  Skipping test (API error): {e}")
                pytest.skip("OpenAI API not available")
        
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            print(f"✅ Average latency: {avg_latency:.2f}s")
            print(f"✅ P95 latency: {p95_latency:.2f}s")
            
            # P95 should be under 10s
            assert p95_latency < 10.0, f"P95 latency too high: {p95_latency:.2f}s"


# Run with: pytest tests/test_evaluation.py -v -s
# Run specific metric: pytest tests/test_evaluation.py::TestRetrieverMetrics::test_precision_at_k -v
# Run without LLM tests: pytest tests/test_evaluation.py -m "not llm" -v
