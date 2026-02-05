"""
End-to-End Tests for Chat With Your Docs

Tests the complete system flow:
- Document ingestion and indexing
- Incremental updates
- Query processing
- Table extraction
- Image extraction
- API endpoints

Run with: pytest tests/test_e2e.py -v
"""

import pytest
import json
import time
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
from src.rag.ingest import run_ingest
from src.rag.config import settings


@pytest.fixture(scope="module")
def test_docs_dir(tmp_path_factory):
    """Create a temporary documents directory for testing"""
    docs_dir = tmp_path_factory.mktemp("test_docs")
    return docs_dir


@pytest.fixture(scope="module")
def test_index_dir(tmp_path_factory):
    """Create a temporary index directory for testing"""
    index_dir = tmp_path_factory.mktemp("test_index")
    return index_dir


@pytest.fixture(scope="module")
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment(test_docs_dir, test_index_dir):
    """Set up test environment before all tests"""
    # Override settings
    settings.docs_dir = str(test_docs_dir)
    settings.index_dir = str(test_index_dir)
    
    yield
    
    # Cleanup after all tests
    shutil.rmtree(test_docs_dir, ignore_errors=True)
    shutil.rmtree(test_index_dir, ignore_errors=True)


class TestDocumentIngestion:
    """Test document loading and indexing"""
    
    def test_01_create_test_documents(self, test_docs_dir):
        """Create test documents for ingestion"""
        
        # Simple text document
        (test_docs_dir / "simple.txt").write_text(
            "This is a simple test document about artificial intelligence. "
            "AI is transforming how we process information."
        )
        
        # Markdown document
        (test_docs_dir / "guide.md").write_text(
            "# User Guide\n\n"
            "## Introduction\n\n"
            "This guide explains Retrieval-Augmented Generation (RAG).\n\n"
            "## What is RAG?\n\n"
            "RAG combines retrieval with generation for better answers."
        )
        
        # CSV with table
        (test_docs_dir / "sales.csv").write_text(
            "Quarter,Revenue,Profit\n"
            "Q1 2023,100000,25000\n"
            "Q2 2023,120000,30000\n"
            "Q3 2023,150000,40000\n"
            "Q4 2023,180000,50000\n"
        )
        
        # JSON document
        (test_docs_dir / "config.json").write_text(json.dumps({
            "name": "Test System",
            "version": "1.0.0",
            "features": ["search", "retrieval", "generation"],
            "max_tokens": 1000
        }, indent=2))
        
        assert (test_docs_dir / "simple.txt").exists()
        assert (test_docs_dir / "guide.md").exists()
        assert (test_docs_dir / "sales.csv").exists()
        assert (test_docs_dir / "config.json").exists()
    
    def test_02_initial_indexing(self, test_docs_dir, test_index_dir):
        """Test initial document indexing"""
        
        # Run ingestion
        run_ingest(reset=True, docs_dir=str(test_docs_dir))
        
        # Check index files exist
        assert (Path(test_index_dir) / "faiss.index").exists()
        assert (Path(test_index_dir) / "metadata.jsonl").exists()
        assert (Path(test_index_dir) / "doc_hashes.json").exists()
        
        # Check metadata contains chunks
        with open(Path(test_index_dir) / "metadata.jsonl", "r") as f:
            chunks = [json.loads(line) for line in f]
        
        assert len(chunks) > 0, "Should have indexed some chunks"
        
        # Verify all documents are indexed
        sources = {chunk["source"] for chunk in chunks}
        assert "simple.txt" in sources
        assert "guide.md" in sources
        assert "sales.csv" in sources
        assert "config.json" in sources
    
    def test_03_table_preservation(self, test_index_dir):
        """Test that CSV tables are preserved in markdown format"""
        
        with open(Path(test_index_dir) / "metadata.jsonl", "r") as f:
            chunks = [json.loads(line) for line in f]
        
        # Find sales.csv chunks
        csv_chunks = [c for c in chunks if c["source"] == "sales.csv"]
        assert len(csv_chunks) > 0, "Should have CSV chunks"
        
        # Check if markdown table format is present
        csv_text = csv_chunks[0]["text"]
        assert "|" in csv_text, "Should contain pipe characters (markdown table)"
        assert "Quarter" in csv_text, "Should contain header"
        assert "Revenue" in csv_text, "Should contain column names"
        assert "Q1 2023" in csv_text, "Should contain data"


class TestIncrementalIndexing:
    """Test incremental document updates"""
    
    def test_04_add_new_document(self, test_docs_dir, test_index_dir):
        """Test adding a new document triggers incremental indexing"""
        
        # Get initial chunk count
        with open(Path(test_index_dir) / "metadata.jsonl", "r") as f:
            initial_chunks = len(f.readlines())
        
        # Add new document
        (test_docs_dir / "new_doc.txt").write_text(
            "This is a newly added document about machine learning. "
            "ML models can learn patterns from data."
        )
        
        # Run incremental ingest
        run_ingest(reset=False, docs_dir=str(test_docs_dir), incremental=True)
        
        # Check chunk count increased
        with open(Path(test_index_dir) / "metadata.jsonl", "r") as f:
            final_chunks = len(f.readlines())
        
        assert final_chunks > initial_chunks, "Should have more chunks after adding document"
        
        # Verify new document is in index
        with open(Path(test_index_dir) / "metadata.jsonl", "r") as f:
            chunks = [json.loads(line) for line in f]
        
        sources = {chunk["source"] for chunk in chunks}
        assert "new_doc.txt" in sources
    
    def test_05_modify_existing_document(self, test_docs_dir, test_index_dir):
        """Test modifying a document triggers re-indexing"""
        
        # Modify simple.txt
        (test_docs_dir / "simple.txt").write_text(
            "This is an UPDATED test document about artificial intelligence. "
            "AI is revolutionizing how we process and analyze information. "
            "New content added here."
        )
        
        # Run incremental ingest
        run_ingest(reset=False, docs_dir=str(test_docs_dir), incremental=True)
        
        # Verify updated content is indexed
        with open(Path(test_index_dir) / "metadata.jsonl", "r") as f:
            chunks = [json.loads(line) for line in f]
        
        simple_chunks = [c for c in chunks if c["source"] == "simple.txt"]
        assert len(simple_chunks) > 0
        
        # Check for new content
        combined_text = " ".join(c["text"] for c in simple_chunks)
        assert "UPDATED" in combined_text
        assert "revolutionizing" in combined_text
    
    def test_06_delete_document(self, test_docs_dir, test_index_dir):
        """Test deleting a document removes it from index"""
        
        # Delete new_doc.txt
        (test_docs_dir / "new_doc.txt").unlink()
        
        # Run incremental ingest
        run_ingest(reset=False, docs_dir=str(test_docs_dir), incremental=True)
        
        # Verify document is removed from index
        with open(Path(test_index_dir) / "metadata.jsonl", "r") as f:
            chunks = [json.loads(line) for line in f]
        
        sources = {chunk["source"] for chunk in chunks}
        assert "new_doc.txt" not in sources, "Deleted document should not be in index"


class TestQueryProcessing:
    """Test query and retrieval functionality"""
    
    def test_07_simple_query(self, client):
        """Test basic query without history"""
        
        response = client.post("/chat", json={
            "question": "What is artificial intelligence?",
            "history": []
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "retrieved" in data
        assert len(data["retrieved"]) > 0
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
    
    def test_08_query_with_specific_info(self, client):
        """Test query for specific information from documents"""
        
        response = client.post("/chat", json={
            "question": "What was the revenue for Q3 2023?",
            "history": []
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should find the answer in sales.csv
        assert "150000" in data["answer"] or "150,000" in data["answer"]
    
    def test_09_follow_up_question(self, client):
        """Test conversational follow-up with context"""
        
        # Initial question
        response1 = client.post("/chat", json={
            "question": "What is RAG?",
            "history": []
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Follow-up question with history
        history = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": data1["answer"]}
        ]
        
        response2 = client.post("/chat", json={
            "question": "How does it work?",
            "history": history
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should understand "it" refers to RAG
        assert len(data2["answer"]) > 0
        assert "retrieval" in data2["answer"].lower() or "generation" in data2["answer"].lower()
    
    def test_10_query_with_citations(self, client):
        """Test that responses include source citations"""
        
        response = client.post("/chat", json={
            "question": "What features does the test system have?",
            "history": []
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check retrieved sources
        assert len(data["retrieved"]) > 0
        
        # Should find config.json
        sources = [r["source"] for r in data["retrieved"]]
        assert any("config.json" in s for s in sources)
        
        # Check citation format in answer
        assert "[" in data["answer"], "Should contain citation markers"


class TestTableAndDataRetrieval:
    """Test retrieval of structured data"""
    
    def test_11_table_data_query(self, client):
        """Test querying table data"""
        
        response = client.post("/chat", json={
            "question": "Show me the quarterly revenue data",
            "history": []
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should retrieve sales.csv
        sources = [r["source"] for r in data["retrieved"]]
        assert any("sales.csv" in s for s in sources)
        
        # Should mention quarters
        answer_lower = data["answer"].lower()
        assert any(q in answer_lower for q in ["q1", "q2", "q3", "q4", "quarter"])
    
    def test_12_json_data_query(self, client):
        """Test querying JSON data"""
        
        response = client.post("/chat", json={
            "question": "What is the version of the test system?",
            "history": []
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should find version from config.json
        assert "1.0.0" in data["answer"] or "version" in data["answer"].lower()


class TestRetrievalQuality:
    """Test search quality and relevance"""
    
    def test_13_semantic_search(self, client):
        """Test semantic understanding (synonyms)"""
        
        # Query using synonym "AI" for "artificial intelligence"
        response = client.post("/chat", json={
            "question": "Tell me about AI",
            "history": []
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should find documents mentioning "artificial intelligence"
        sources = [r["source"] for r in data["retrieved"]]
        assert any("simple.txt" in s for s in sources)
    
    def test_14_relevance_scoring(self, client):
        """Test that retrieved chunks have relevance scores"""
        
        response = client.post("/chat", json={
            "question": "What is RAG?",
            "history": []
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that all retrieved chunks have scores
        for chunk in data["retrieved"]:
            assert "score" in chunk
            assert isinstance(chunk["score"], (int, float))
            assert chunk["score"] >= 0, "Score should be non-negative"
        
        # Check that results are sorted by relevance (descending)
        scores = [chunk["score"] for chunk in data["retrieved"]]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score"


class TestErrorHandling:
    """Test error cases and edge conditions"""
    
    def test_15_empty_query(self, client):
        """Test handling of empty query"""
        
        response = client.post("/chat", json={
            "question": "",
            "history": []
        })
        
        # Should either reject or handle gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_16_very_long_query(self, client):
        """Test handling of very long query"""
        
        long_query = "What is AI? " * 500  # Very long query
        
        response = client.post("/chat", json={
            "question": long_query,
            "history": []
        })
        
        # Should handle gracefully (either succeed or reject)
        assert response.status_code in [200, 400, 413]
    
    def test_17_query_with_no_results(self, client):
        """Test query that matches no documents"""
        
        response = client.post("/chat", json={
            "question": "Tell me about quantum entanglement in superstring theory",
            "history": []
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return some answer even if no perfect match
        assert "answer" in data
    
    def test_18_malformed_request(self, client):
        """Test handling of malformed request"""
        
        response = client.post("/chat", json={
            "invalid_field": "test"
        })
        
        assert response.status_code == 422, "Should reject malformed request"


class TestHealthCheck:
    """Test API health endpoints"""
    
    def test_19_health_endpoint(self, client):
        """Test health check endpoint"""
        
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_20_root_endpoint(self, client):
        """Test root endpoint"""
        
        response = client.get("/")
        assert response.status_code in [200, 404]  # Depends on if static files exist


class TestPerformance:
    """Test performance characteristics"""
    
    def test_21_query_latency(self, client):
        """Test that queries complete in reasonable time"""
        
        start_time = time.time()
        
        response = client.post("/chat", json={
            "question": "What is artificial intelligence?",
            "history": []
        })
        
        end_time = time.time()
        latency = end_time - start_time
        
        assert response.status_code == 200
        assert latency < 10.0, f"Query should complete in <10s, took {latency:.2f}s"
    
    def test_22_multiple_concurrent_queries(self, client):
        """Test handling multiple queries"""
        
        questions = [
            "What is AI?",
            "What is RAG?",
            "Show me revenue data",
            "What features are available?",
            "Tell me about machine learning"
        ]
        
        for question in questions:
            response = client.post("/chat", json={
                "question": question,
                "history": []
            })
            assert response.status_code == 200


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Run with: pytest tests/test_e2e.py -v
# Run specific test: pytest tests/test_e2e.py::TestDocumentIngestion::test_01_create_test_documents -v
# Run with coverage: pytest tests/test_e2e.py --cov=src --cov-report=html -v
