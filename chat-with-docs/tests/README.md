# End-to-End Testing Guide

## Overview

Comprehensive E2E test suite covering:
- Document ingestion (initial and incremental)
- Query processing and retrieval
- Table and structured data handling
- API endpoints
- Error handling
- Performance validation

## Installation

Install test dependencies:
```bash
pip install -e .
```

This installs pytest and httpx (FastAPI test client).

## Running Tests

### Run all tests:
```bash
pytest tests/test_e2e.py -v
```

### Run specific test class:
```bash
pytest tests/test_e2e.py::TestDocumentIngestion -v
```

### Run specific test:
```bash
pytest tests/test_e2e.py::TestQueryProcessing::test_07_simple_query -v
```

### Run with coverage:
```bash
pytest tests/test_e2e.py --cov=src --cov-report=html -v
```

### Run excluding slow tests:
```bash
pytest tests/test_e2e.py -m "not slow" -v
```

## Test Structure

Tests are organized in sequential classes:

### 0. RAG Evaluation Tests (test_evaluation.py)
**Purpose**: Measure RAG system quality with standard metrics

- **TestChunkQuality** - Semantic coherence, information density, boundary quality
- **TestRetrieverMetrics** - Precision@K, Recall@K, MRR, NDCG@K
- **TestGeneratorQuality** - Faithfulness, relevance, concept coverage
- **TestEndToEndRAG** - Answer correctness, latency benchmarks

**Metrics Used**:
- Precision@K: Fraction of top-K results that are relevant
- Recall@K: Fraction of relevant documents in top-K
- MRR: Average reciprocal rank of first relevant result
- NDCG@K: Normalized discounted cumulative gain (position-aware)
- Faithfulness: Answer grounding in retrieved context
- Relevance: Semantic similarity between question and answer

**Ground Truth Dataset**: 5 AI/ML questions with known relevant documents

### 1. TestDocumentIngestion (tests 01-03)
- Creates test documents (txt, md, csv, json)
- Tests initial indexing
- Verifies table preservation in markdown format

### 2. TestIncrementalIndexing (tests 04-06)
- Tests adding new documents
- Tests modifying existing documents
- Tests deleting documents
- Verifies incremental updates are faster

### 3. TestQueryProcessing (tests 07-10)
- Tests simple queries
- Tests specific information retrieval
- Tests follow-up questions with context
- Tests citation and source attribution

### 4. TestTableAndDataRetrieval (tests 11-12)
- Tests querying table data (CSV)
- Tests querying structured data (JSON)

### 5. TestRetrievalQuality (tests 13-14)
- Tests semantic search (synonyms)
- Tests relevance scoring and ranking

### 6. TestErrorHandling (tests 15-18)
- Tests empty queries
- Tests very long queries
- Tests queries with no results
- Tests malformed requests

### 7. TestHealthCheck (tests 19-20)
- Tests health endpoint
- Tests root endpoint

### 8. TestPerformance (tests 21-22)
- Tests query latency (<10s)
- Tests concurrent query handling

## Test Fixtures

- `test_docs_dir`: Temporary directory for test documents
- `test_index_dir`: Temporary directory for test index
- `client`: FastAPI TestClient for API testing
- `setup_test_environment`: Configures settings and cleanup

## Expected Results

All tests should pass with:
- ✅ 22 tests total
- ✅ Document ingestion working
- ✅ Incremental updates functional
- ✅ Query processing accurate
- ✅ Table preservation correct
- ✅ Error handling robust
- ✅ Performance within limits

## Test Data

Tests create temporary documents:
- `simple.txt` - Plain text about AI
- `guide.md` - Markdown with headers about RAG
- `sales.csv` - Table with quarterly revenue
- `config.json` - JSON configuration
- `new_doc.txt` - Added during incremental test

All test data is cleaned up automatically.

## Troubleshooting

### Tests fail with "No module named 'src'"
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/test_e2e.py -v
```

### Tests fail with "No OpenAI API key"
Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
pytest tests/test_e2e.py -v
```

### Tests timeout
Increase timeout in pytest.ini or skip performance tests:
```bash
pytest tests/test_e2e.py -m "not slow" -v
```

## Continuous Integration

Add to CI/CD pipeline:
```yaml
- name: Run E2E Tests
  run: |
    pip install -e .
    pytest tests/test_e2e.py -v --cov=src --cov-report=xml
```

## Coverage Goals

Target coverage: >80% for:
- `src/rag/ingest.py` - Ingestion logic
- `src/rag/vectorstore.py` - Search functionality
- `src/rag/chat.py` - Query processing
- `src/api/main.py` - API endpoints
