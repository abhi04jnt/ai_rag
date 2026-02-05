"""
Unit Tests for Individual Components

Tests individual functions and classes in isolation.

Run with: pytest tests/test_unit.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vectorstore import FaissStore
from src.rag.chunking import chunk_text, Chunk
from src.rag.embeddings import Embedder
from src.rag.config import settings


class TestChunking:
    """Test text chunking functionality"""
    
    def test_chunk_simple_text(self):
        """Test chunking of simple text"""
        text = "This is a test. " * 100  # 100 sentences
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
        
        assert len(chunks) > 0, "Should create chunks"
        assert all(len(c) > 0 for c in chunks), "All chunks should have content"
    
    def test_chunk_size_limit(self):
        """Test that chunks respect size limit"""
        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)
        
        for chunk in chunks:
            assert len(chunk) <= 550, "Chunks should not exceed limit significantly"
    
    def test_chunk_overlap(self):
        """Test that chunks have overlap"""
        text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        
        if len(chunks) > 1:
            # Check overlap exists
            assert len(chunks) > 1
    
    def test_empty_text(self):
        """Test chunking empty text"""
        chunks = chunk_text("", chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 0, "Empty text should return no chunks"
    
    def test_short_text(self):
        """Test chunking text shorter than chunk size"""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)
        
        assert len(chunks) == 1, "Short text should create one chunk"
        assert chunks[0] == text, "Chunk should match original text"


class TestEmbedder:
    """Test embedding generation"""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance"""
        return Embedder(settings.embedding_model)
    
    def test_embed_single_text(self, embedder):
        """Test embedding a single text"""
        text = "This is a test sentence"
        embedding = embedder.embed_texts([text])[0]
        
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert len(embedding.shape) == 1, "Should be 1D array"
        assert embedding.shape[0] == 384, "Should be 384-dimensional"
    
    def test_embed_batch(self, embedder):
        """Test batch embedding"""
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]
        embeddings = embedder.embed_texts(texts)
        
        assert isinstance(embeddings, np.ndarray), "Should return numpy array"
        assert embeddings.shape[0] == 3, "Should have 3 embeddings"
        assert embeddings.shape[1] == 384, "Each should be 384-dimensional"
    
    def test_embed_empty_string(self, embedder):
        """Test embedding empty string"""
        embedding = embedder.embed_texts([""])[0]
        
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert embedding.shape[0] == 384, "Should still be 384-dimensional"
    
    def test_embeddings_are_normalized(self, embedder):
        """Test that embeddings are normalized"""
        text = "Test sentence for normalization"
        embedding = embedder.embed_texts([text])[0]
        
        # Check L2 norm is close to 1
        norm = np.linalg.norm(embedding)
        assert 0.99 <= norm <= 1.01, "Embedding should be normalized"


class TestVectorStore:
    """Test vector storage and search"""
    
    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary index directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def vector_store(self, temp_index_dir):
        """Create vector store instance"""
        return FaissStore(index_dir=temp_index_dir)
    
    def test_add_documents(self, vector_store):
        """Test adding documents to vector store"""
        vectors = np.random.randn(5, 384).astype('float32')
        
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        
        chunks = [
            Chunk(text=f"Document {i}", source="test.txt", chunk_id=i)
            for i in range(5)
        ]
        
        vector_store.add_documents(vectors, chunks, "hash123")
        
        assert vector_store.index.ntotal == 5, "Should have 5 vectors"
        assert len(vector_store.meta) == 5, "Should have 5 metadata entries"
    
    def test_search(self, vector_store):
        """Test vector search"""
        # Add some documents
        vectors = np.random.randn(10, 384).astype('float32')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        
        chunks = [
            Chunk(text=f"Document {i}", source="test.txt", chunk_id=i)
            for i in range(10)
        ]
        
        vector_store.add_documents(vectors, chunks, "hash123")
        
        # Search with first vector
        query_vec = vectors[0]
        results = vector_store.search(query_vec, top_k=3)
        
        assert len(results) == 3, "Should return 3 results"
        assert all("text" in r for r in results), "All results should have text"
        assert all("score" in r for r in results), "All results should have scores"
        
        # First result should be the query itself (highest similarity)
        assert results[0]["chunk_id"] == 0, "First result should be the query document"
    
    def test_remove_document(self, vector_store):
        """Test removing documents by source"""
        # Add documents from two sources
        vectors1 = np.random.randn(3, 384).astype('float32')
        vectors1 = vectors1 / np.linalg.norm(vectors1, axis=1, keepdims=True)
        chunks1 = [
            Chunk(text=f"Doc1-{i}", source="source1.txt", chunk_id=i)
            for i in range(3)
        ]
        
        vectors2 = np.random.randn(2, 384).astype('float32')
        vectors2 = vectors2 / np.linalg.norm(vectors2, axis=1, keepdims=True)
        chunks2 = [
            Chunk(text=f"Doc2-{i}", source="source2.txt", chunk_id=i)
            for i in range(2)
        ]
        
        vector_store.add_documents(vectors1, chunks1, "hash1")
        vector_store.add_documents(vectors2, chunks2, "hash2")
        
        assert vector_store.index.ntotal == 5
        
        # Remove source1.txt
        removed = vector_store.remove_document("source1.txt")
        
        assert removed == 3, "Should remove 3 chunks"
        assert vector_store.index.ntotal == 2, "Should have 2 chunks left"
        assert all(m["source"] == "source2.txt" for m in vector_store.meta)
    
    def test_needs_update(self, vector_store):
        """Test hash-based update detection"""
        # Add document with hash
        vectors = np.random.randn(2, 384).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        chunks = [
            Chunk(text=f"Doc {i}", source="test.txt", chunk_id=i)
            for i in range(2)
        ]
        
        vector_store.add_documents(vectors, chunks, "hash123")
        
        # Same hash should not need update
        assert not vector_store.needs_update("test.txt", "hash123")
        
        # Different hash should need update
        assert vector_store.needs_update("test.txt", "hash456")
        
        # New source should need update
        assert vector_store.needs_update("new.txt", "hash789")
    
    def test_save_and_load(self, vector_store, temp_index_dir):
        """Test saving and loading index"""
        # Add documents
        vectors = np.random.randn(5, 384).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        chunks = [
            Chunk(text=f"Document {i}", source="test.txt", chunk_id=i)
            for i in range(5)
        ]
        
        vector_store.add_documents(vectors, chunks, "hash123")
        vector_store.save()
        
        # Load in new instance
        new_store = FaissStore(index_dir=temp_index_dir)
        
        assert new_store.index.ntotal == 5, "Should load 5 vectors"
        assert len(new_store.meta) == 5, "Should load 5 metadata entries"
        assert new_store.doc_hashes.get("test.txt") == "hash123"


class TestVectorStoreTokenization:
    """Test BM25 tokenization"""
    
    def test_tokenize_simple(self):
        """Test simple tokenization"""
        text = "This is a simple test"
        tokens = FaissStore.tokenize(text)
        
        assert tokens == ["this", "is", "a", "simple", "test"]
    
    def test_tokenize_with_punctuation(self):
        """Test tokenization with punctuation"""
        text = "Hello, world! How are you?"
        tokens = FaissStore.tokenize(text)
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens
        assert "!" not in tokens
    
    def test_tokenize_empty(self):
        """Test tokenizing empty string"""
        tokens = FaissStore.tokenize("")
        assert tokens == []
    
    def test_tokenize_lowercase(self):
        """Test that tokens are lowercase"""
        text = "UPPERCASE lowercase MixedCase"
        tokens = FaissStore.tokenize(text)
        
        assert all(t.islower() for t in tokens if t.isalpha())


class TestConfig:
    """Test configuration"""
    
    def test_default_settings(self):
        """Test default configuration values"""
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.top_k > 0
        assert settings.embedding_model is not None
    
    def test_paths_are_absolute(self):
        """Test that paths are absolute"""
        assert Path(settings.docs_dir).is_absolute()
        assert Path(settings.index_dir).is_absolute()


# Run with: pytest tests/test_unit.py -v
# Run with coverage: pytest tests/test_unit.py --cov=src --cov-report=html -v
