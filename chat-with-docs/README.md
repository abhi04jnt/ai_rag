# Chat With Your Docs (RAG)

## Overview
This project implements a production-ready conversational RAG (Retrieval-Augmented Generation) system that answers questions using local document collections with **multi-format support**, **automatic indexing**, **follow-up conversation handling**, and **visual content extraction**.

**Key Features:**
- 🗂️ **Multi-format Support**: 15+ formats including PDF, DOCX, PPTX, XLSX, HTML, XML, EML, CSV, JSON, Markdown, TXT
- 🖼️ **Image Extraction**: Automatically extracts and indexes diagrams, charts, and images from PDFs
- 💬 **Conversational AI**: LLM-based query reformulation for natural follow-up questions
- 🔄 **Incremental Indexing**: Hash-based change detection - only re-indexes modified files
- 🔍 **Semantic Search**: Pure vector search optimized for understanding related concepts and synonyms
- 📊 **Smart Retrieval**: Visual query detection prioritizes image-rich content
- 📋 **Table Preservation**: Maintains table structure in markdown format for better comprehension
- 🚀 **FastAPI Backend**: REST API with auto-reload during development
- 🔒 **Privacy-First**: Optional content safety filters for compliance

**RAG Pipeline:**
1. **Ingest** → Load documents from `data/` directory recursively
2. **Parse** → Extract text and images using format-specific parsers
3. **Chunk** → Split into overlapping chunks (1000 tokens, 200 overlap)
4. **Embed** → Generate 384-dim vectors using sentence-transformers
5. **Index** → Store in FAISS vector database with metadata
6. **Query** → Reformulate follow-ups, search semantically, return top-k chunks
7. **Generate** → LLM creates grounded answers with citations and images

**Current Index Stats:**
- 📚 38,863 chunks from 201 documents
- 🖼️ 70+ extracted images from PDFs
- 🗄️ 159MB total index size (57MB vectors + 102MB metadata)

## 🎯 Architecture

```
User Query → Safety Filter → Query Reformulation (LLM)
                                    ↓
                          Semantic Search (FAISS)
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Text Chunks                      Image Chunks
                    └───────────────┬───────────────┘
                                    ↓
                            LLM Generation
                                    ↓
                    Answer + Citations + Images
```

## 🔒 Privacy & Compliance Features (Optional)

Configurable safety measures for regulated environments:
- **HIPAA** (Health Insurance Portability and Accountability Act)
- **GDPR** (General Data Protection Regulation)  
- **AI Acts** and Ethical AI Guidelines

### Protected Content Detection
The system can detect and refuse to process:
- ❌ **PII**: SSNs, phone numbers, emails, credit cards, passports
- ❌ **PHI**: Medical records, diagnoses, prescriptions
- ❌ **Sensitive Data**: Biometric, genetic, financial records
- ❌ **Inappropriate Content**: Explicit material

**Note**: Safety filters can be adjusted in `src/rag/prompts.py` based on your use case.

## 🏗️ Design Decisions & Trade-offs

### Retrieval store: FAISS + JSONL metadata
- **Why**: Fast local similarity search (IndexFlatIP), no external dependencies, trivial deployment
- **✅ Pros**: 
  - Sub-100ms search on 38K chunks
  - Simple file-based storage (faiss.index + metadata.jsonl)
  - No database setup or maintenance
  - Easy backup and version control
- **❌ Cons**: 
  - No native metadata filtering (must post-filter in Python)
  - Manual metadata management (separate JSONL file)
  - RAM-resident index (38K chunks = ~57MB)
  - No distributed querying for massive datasets
- **Alternative**: Qdrant/Weaviate for advanced filtering, Pinecone for managed service

### Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- **Why**: Lightweight (80MB model), fast inference (~50ms/query), runs locally, widely adopted baseline
- **✅ Pros**: 
  - No API costs or rate limits
  - 384-dimensional vectors (smaller index)
  - Good general-domain performance
  - Reproducible results
- **❌ Cons**: 
  - Lower retrieval quality vs larger models (e.g., text-embedding-3-large)
  - Not optimized for domain-specific language
  - Limited context window (256 tokens)
  - No fine-tuning out-of-the-box
- **Alternative**: OpenAI text-embedding-3-small (1536-dim, higher quality), custom fine-tuned models

### Chunking: Fixed-size with overlap (1000 tokens, 200 overlap)
- **Why**: Predictable, simple, language-agnostic, preserves some context across boundaries
- **✅ Pros**: 
  - Works for all document types
  - Prevents context loss at boundaries (200-token overlap)
  - Consistent chunk sizes for embedding
  - Easy to implement and debug
- **❌ Cons**: 
  - Can split semantic units (paragraphs, sections)
  - No awareness of document structure (headings, lists)
  - Fixed size may be suboptimal for different content types
  - Duplicate content at boundaries increases index size
- **Alternative**: Structure-aware chunking (by section/paragraph), semantic chunking, recursive splitting

### LLM Provider: OpenAI (GPT-4o-mini) with Ollama fallback
- **Why**: GPT-4o-mini offers best quality-to-cost ratio, Ollama enables offline testing
- **✅ Pros**: 
  - High-quality responses with good instruction following
  - Fast inference (~1-2s response time)
  - Provider abstraction supports switching
  - Ollama for local/offline deployments
- **❌ Cons**: 
  - OpenAI requires API key and incurs costs (~$0.15/1M input tokens)
  - API latency (100-200ms network overhead)
  - Rate limits on OpenAI API
  - Local models (Ollama) have lower quality
- **Alternative**: Anthropic Claude (better reasoning), Llama 3.2 (free local), GPT-4 (highest quality)

### Conversational Context: LLM-based Query Reformulation
- **Why**: Handles ambiguous follow-ups ("share me the architecture diagram" after "Define RAG")
- **✅ Pros**: 
  - Semantic understanding of conversation flow
  - Handles pronouns and references ("it", "that", "this")
  - Works across different question patterns
  - No manual rule crafting
- **❌ Cons**: 
  - Extra LLM call adds 500-800ms latency
  - Costs ~200 tokens per follow-up question
  - Can misinterpret context in complex conversations
  - No explicit entity tracking
- **Alternative**: Regex patterns (faster but brittle), conversation memory vectors, explicit state tracking

### Hybrid Search: BM25 + Vector with RRF Fusion
- **Why**: Combines keyword matching (BM25) with semantic search (vectors) for better recall
- **✅ Pros**: 
  - Finds exact term matches that vectors might miss
  - Better for technical terms, acronyms, and names
  - RRF fusion is robust to different score scales
  - Improves recall by 15-30% over pure vector search (for keyword-heavy queries)
  - No extra API calls (computed locally)
- **❌ Cons**: 
  - Adds ~20-30ms to search latency
  - Requires storing tokenized corpus (~10% more disk space)
  - Simple tokenization (split by whitespace) may not handle all languages
  - Requires rebuilding BM25 index on updates
  - May reduce performance for semantic queries if BM25 weight is too high
- **Alternative**: Pure vector search (faster, better for semantic queries), learned sparse retrieval (SPLADE)
- **Note**: Currently disabled by default - pure semantic search works better for most queries. Can be enabled by setting `hybrid=True` in [chat.py](src/rag/chat.py)

### Document Parsing: Unstructured + Custom Loaders
- **Why**: Unified interface for 15+ formats, custom PDF image extraction
- **✅ Pros**: 
  - Single codebase for all formats
  - Extracts text, tables, images
  - Handles encoding detection (UTF-8, UTF-16, etc.)
  - Format-specific optimizations (DOCX styles, Excel sheets)
- **❌ Cons**: 
  - PDF layout extraction is imperfect (reading order issues)
  - No OCR for scanned PDFs
  - Table structure often lost (converted to text)
  - Image extraction limited to /XObject resources (no inline images)
- **Alternative**: Apache Tika (more formats), LlamaParse (better layout), Docling (production-grade)

### Image Handling: Base64 in Metadata
- **Why**: Simple storage, no external file management, works with JSON serialization
- **✅ Pros**: 
  - Self-contained metadata (no broken image links)
  - Easy to transfer and backup
  - Direct embedding in API responses
  - Preserves image-chunk relationships
- **❌ Cons**: 
  - Large metadata file (102MB for 70 images)
  - Base64 encoding increases size by ~33%
  - Must decode for display
  - No image deduplication
- **Alternative**: Separate image store (S3/filesystem), image hashing for deduplication

### Auto-indexing: Watchdog File Monitor
- **Why**: Automatically detects changes in `data/` and re-indexes
- **✅ Pros**: 
  - No manual re-indexing needed
  - Watches subdirectories recursively
  - Debounced updates (waits for writes to complete)
  - Background processing
- **❌ Cons**: 
  - Full re-index on any change (no incremental updates)
  - No change tracking (can't see what changed)
  - Race conditions if files change during indexing
  - Extra CPU/memory overhead
- **Alternative**: Hash-based change detection, incremental updates, manual trigger

### Citations: LLM-instructed format [source:file:chunk]
- **Why**: Simple, grounded, easy to parse, traces answers back to sources
- **✅ Pros**: 
  - Improves answer faithfulness
  - Users can verify information
  - Easy to render as links
  - LLM generally follows format
- **❌ Cons**: 
  - Not automatically validated (LLM might hallucinate)
  - No fine-grained sentence-level attribution
  - Requires post-processing to make clickable
  - Can clutter short answers
- **Alternative**: Automatic citation parsing, in-line references, confidence scores

## 🚀 How to Run

### Option 1: Local Development

#### 1) Install Dependencies
```bash
# Python 3.10+ required (tested on 3.13)
pip install -e .
```

**Key Dependencies:**
- `fastapi` + `uvicorn` - API server
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Vector search
- `openai` - LLM provider
- `pypdf` - PDF parsing with image extraction
- `python-docx`, `python-pptx`, `openpyxl` - Office formats
- `beautifulsoup4` - HTML/XML parsing
- `watchdog` - Auto-indexing

#### 2) Configure Environment
Create `.env` file:
```bash
# Required
OPENAI_API_KEY=your-key-here

# Optional (defaults shown)
DOCS_DIR=./data
TOP_K=5
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### 3) Add Documents
Place files in `data/` directory (recursive scanning):
```
data/
  docs/
    report.pdf
    guide.docx
  example-docs/
    spreadsheet.xlsx
    presentation.pptx
```

**Supported Formats:**
PDF, DOCX, PPTX, XLSX, HTML, XML, EML, MSG, CSV, TSV, JSON, YAML, MD, TXT, GO

#### 4) Start Server (Auto-indexes on first run)
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Server will:
- ✅ Check for existing index
- ✅ Auto-index if missing (one-time)
- ✅ Start file watcher for changes
- ✅ Extract images from PDFs
- 🌐 API available at `http://localhost:8000`

#### 5) Query via API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "history": []
  }'
```

### Option 2: Docker Deployment 🐳

#### Quick Start
```bash
# 1. Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env

# 2. Place documents in data/docs/
mkdir -p data/docs
cp your-documents/* data/docs/

# 3. Start with Docker Compose
docker-compose up -d

# 4. Check logs
docker-compose logs -f

# 5. Stop services
docker-compose down
```

#### Using the Deployment Script
```bash
# Make script executable
chmod +x deploy-docker.sh

# Build image
./deploy-docker.sh build

# Start services
./deploy-docker.sh up

# View logs
./deploy-docker.sh logs

# Restart
./deploy-docker.sh restart

# Stop
./deploy-docker.sh down

# Clean up everything
./deploy-docker.sh clean
```

#### Manual Docker Commands
```bash
# Build image
docker build -t chat-with-docs:latest .

# Run container
docker run -d \
  --name chat-with-docs \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-key" \
  -v $(pwd)/data/docs:/app/data/docs:ro \
  -v chat-index:/app/data/index \
  chat-with-docs:latest

# View logs
docker logs -f chat-with-docs

# Stop container
docker stop chat-with-docs
docker rm chat-with-docs
```

#### Docker Configuration

**Environment Variables:**
- `OPENAI_API_KEY` - Required for LLM
- `OPENAI_MODEL` - Default: `gpt-4o-mini`
- `EMBEDDING_MODEL` - Default: `sentence-transformers/all-MiniLM-L6-v2`
- `TOP_K` - Default: `5`
- `CHUNK_SIZE` - Default: `1200`
- `CHUNK_OVERLAP` - Default: `200`

**Volumes:**
- `./data/docs:/app/data/docs:ro` - Documents (read-only)
- `chat-index:/app/data/index` - Persistent index storage

**Ports:**
- `8000` - FastAPI server

#### Health Check
```bash
# Check container health
docker ps

# Test API endpoint
curl http://localhost:8000/health
```

### API Response Format

#### 5) Query via API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "history": []
  }'
```

**Response:**
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is...",
  "retrieved": [
    {
      "source": "rag-guide.pdf",
      "chunk_id": 15,
      "score": 0.85,
      "text": "...",
      "images": [
        {
          "page": 3,
          "format": "PNG",
          "width": 800,
          "height": 600,
          "data": "base64-encoded-image..."
        }
      ]
    }
  ]
}
```

### 6) Follow-up Questions
```bash
curl -X POST http://localhost:8000/chat \
  -d '{
    "question": "share me the architecture diagram",
    "history": [
      {"role": "user", "content": "What is RAG?"},
      {"role": "assistant", "content": "RAG is..."}
    ]
  }'
```

System automatically reformulates query using conversation context.

### 7) Manual Re-indexing

**Incremental (default - only changed files):**
```bash
python -m src.rag.ingest
```

**Full rebuild (all files):**
```bash
python -m src.rag.ingest --full
```

**Reset and rebuild:**
```bash
python -m src.rag.ingest --reset
```

### 8) Process Unstructured Data (CLI)
```bash
python -m src.rag.process_unstructured_docs data/example-docs/
```

## 📈 Performance Metrics

**Indexing:**
- 201 documents → 38,863 chunks in ~45 seconds (full rebuild)
- Incremental updates: 1-5 seconds per changed document
- PDF image extraction: ~2-3 pages/second
- Index size: 165MB (57MB vectors + 102MB metadata + 6MB BM25)

**Query Latency:**
- Embedding: ~50ms
- Vector search: ~80ms (pure semantic - default)
- Hybrid search (optional): ~100ms (+20ms for BM25)
- LLM generation: ~1-2s
- Query reformulation: +500-800ms (follow-ups only)
- **Total**: ~1.5-3s per query

**Retrieval Quality:**
- TOP_K=5 default (configurable)
- Pure vector search: Best for semantic queries, synonyms, related concepts
- Hybrid search (optional): Better for exact keyword matches
- Visual queries retrieve 2x TOP_K, prioritize image chunks
- Average relevance score: 0.65-0.85 for good queries

## 🔧 Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `DOCS_DIR` | `./data` | Document directory (recursive) |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `OPENAI_API_KEY` | - | Required for GPT models |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |

### Adjusting Safety Filters
Edit `src/rag/prompts.py` SYSTEM_PROMPT to customize compliance rules.

### Chunking Parameters
Edit `src/rag/ingest.py`:
```python
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens
```

## 🛣️ Roadmap & Future Improvements

### High Priority
- [ ] **OCR Support**: Extract text from scanned PDFs using Tesseract
- [ ] **Streaming Responses**: Server-Sent Events for real-time answer generation
- [ ] **Query Rewriting**: Use LLM to expand queries with synonyms and related terms

### Medium Priority
- [ ] **Structure-Aware Chunking**: Split by sections/headings instead of fixed size
- [ ] **Citation Validation**: Parse and verify LLM-generated citations
- [ ] **Query Analytics**: Track popular queries, failed searches, user feedback
- [ ] **Metadata Filtering**: Filter by date, author, document type
- [ ] **Deduplication**: Content-based hashing to avoid indexing duplicates

### Low Priority / Nice-to-Have
- [ ] **Multi-modal Embeddings**: CLIP for joint text-image search
- [ ] **Reranking**: Cross-encoder for better top-k selection
- [ ] **Conversation Memory**: Semantic compression of long histories
- [ ] **Evaluation Suite**: Automated testing for retrieval hit-rate and answer faithfulness
- [ ] **Web UI**: Simple frontend for document upload and chat
- [ ] **Multi-language Support**: Detect and handle non-English documents

## 🐛 Known Issues & Limitations

1. **Image Retrieval**: Visual queries sometimes don't match image-rich chunks due to text-only embeddings (no visual search)
2. **PDF Layout**: Complex layouts with multiple columns can have incorrect reading order
3. **No OCR**: Scanned PDFs with no text layer are skipped
4. **Large Files**: Files >10MB may cause memory issues during parsing
5. **Full Re-index**: Any document change triggers complete re-indexing
6. **Citation Accuracy**: LLM may hallucinate citations not in retrieved chunks
7. **Context Window**: Long conversations may exceed LLM context limits

## 📚 Project Structure

```
chat-with-docs/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI app, endpoints
│   └── rag/
│       ├── __init__.py
│       ├── config.py            # Configuration settings
│       ├── embeddings.py        # Sentence-transformers wrapper
│       ├── vectorstore.py       # FAISS index management
│       ├── ingest.py            # Document processing pipeline
│       ├── unstructured_loader.py  # Multi-format parsers
│       ├── auto_indexer.py      # File watcher for auto-indexing
│       ├── chat.py              # Query reformulation + retrieval
│       ├── llm.py               # LLM provider abstraction
│       └── prompts.py           # System prompts
├── data/
│   ├── docs/                    # Your documents (git-ignored)
│   ├── example-docs/            # Sample test files
│   └── index/                   # Generated index files
│       ├── faiss.index          # Vector database
│       ├── metadata.jsonl       # Chunk metadata + images
│       └── doc_hashes.json      # File change tracking
├── tests/
│   ├── test_unit.py             # Unit tests for components
│   ├── test_e2e.py              # End-to-end integration tests
│   ├── run_tests.sh             # Test runner script
│   └── README.md                # Testing guide
├── pyproject.toml
├── pytest.ini
├── .env
├── README.md
└── DESIGN.md                    # System architecture documentation
```

## 🧪 Testing

Comprehensive test suite with unit and end-to-end tests.

### Quick Start
```bash
# Run all tests
./tests/run_tests.sh all

# Run unit tests only (fast)
./tests/run_tests.sh unit

# Run with coverage
./tests/run_tests.sh coverage
```

### Test Coverage

**Unit Tests** ([tests/test_unit.py](tests/test_unit.py))
- Text chunking (size, overlap, edge cases)
- Embedding generation (single, batch, normalization)
- Vector store operations (add, search, remove)
- BM25 tokenization
- Configuration validation

**E2E Tests** ([tests/test_e2e.py](tests/test_e2e.py))
- Document ingestion (txt, md, csv, json)
- Incremental indexing (add, modify, delete)
- Table preservation in markdown
- Query processing (simple, follow-up, semantic)
- Retrieval quality and relevance scoring
- Error handling (empty queries, no results, malformed requests)
- Performance validation (<10s latency)

**RAG Evaluation Tests** ([tests/test_evaluation.py](tests/test_evaluation.py))
- **Chunk Quality**: Semantic coherence, information density, boundary quality
- **Retriever Metrics**: Precision@K, Recall@K, MRR, NDCG@K
- **Generator Quality**: Faithfulness, relevance, key concept coverage
- **End-to-End**: Answer correctness vs ground truth, latency benchmarks
- **Ground Truth Dataset**: 5 AI/ML Q&A pairs with known relevant docs

### Latest Test Results

```bash
$ pytest tests/test_evaluation.py -v

✅ TestChunkQuality (3/3 passed)
  ✓ test_chunk_semantic_coherence      - Coherence: 0.49 (threshold: >0.45)
  ✓ test_chunk_information_density     - Density: 0.65 (threshold: >0.4)
  ✓ test_chunk_boundary_quality        - Quality: 100% (threshold: >80%)

✅ TestRetrieverMetrics (4/4 passed)
  ✓ test_precision_at_k                - P@1: 0.80, P@3: 0.73, P@5: 0.68
  ✓ test_recall_at_k                   - R@3: 0.80, R@5: 0.87, R@10: 0.93
  ✓ test_mean_reciprocal_rank          - MRR: 0.83 (threshold: >0.4)
  ✓ test_ndcg_at_k                     - NDCG@5: 0.87 (threshold: >0.5)

⏭️  TestGeneratorQuality (3 skipped - requires OPENAI_API_KEY)
  ⊘ test_answer_faithfulness
  ⊘ test_answer_relevance
  ⊘ test_answer_contains_key_concepts

✅ TestEndToEndRAG (1/3 passed, 2 skipped)
  ⊘ test_answer_correctness            - Requires OpenAI API
  ⊘ test_retrieval_impact_on_quality   - Requires OpenAI API
  ✓ test_rag_latency                   - P95: <10s

Summary: 8 passed, 5 skipped, 5 warnings in 15.52s
```

**Set OpenAI API key to run generator/E2E tests:**
```bash
export OPENAI_API_KEY="your-key-here"
pytest tests/test_evaluation.py -v
```

### Manual Testing
```bash
# Install test dependencies
pip install -e .

# Run specific test
pytest tests/test_unit.py::TestChunking::test_chunk_simple_text -v

# Run with detailed output
pytest tests/test_e2e.py -v -s

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

See [tests/README.md](tests/README.md) for detailed testing guide.

---

**check [DESIGN.md](DESIGN.md) for architecture details.
