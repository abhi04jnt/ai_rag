# System Design Document

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Sequence Diagrams](#sequence-diagrams)
6. [Storage Structure](#storage-structure)

---

## System Overview

**Chat With Your Docs** is a RAG (Retrieval-Augmented Generation) system that enables conversational querying over a collection of documents. The system combines semantic search, LLM-based query reformulation, and automatic document indexing to provide accurate, grounded answers with source citations.

### Core Capabilities
- Multi-format document ingestion (15+ formats)
- Incremental indexing with change detection
- Conversational context handling
- Image extraction and retrieval
- Table structure preservation
- Real-time file monitoring

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER / CLIENT                                   │
│                     (HTTP REST API Requests)                             │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FASTAPI SERVER                                     │
│                      (src/api/main.py)                                   │
│                                                                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐    │
│  │   Startup       │  │   /chat          │  │   /health          │    │
│  │   - Check Index │  │   Endpoint       │  │   Endpoint         │    │
│  │   - Init Chat   │  │   - Validate     │  │   - Status Check   │    │
│  │   - Start Watch │  │   - Process      │  │                    │    │
│  └─────────────────┘  └──────────────────┘  └────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CHAT SERVICE                                        │
│                     (src/rag/chat.py)                                    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  1. Safety Filter (PII/PHI detection)                            │  │
│  │  2. Query Reformulation (LLM-based for follow-ups)               │  │
│  │  3. Visual Query Detection (diagram/chart/image keywords)        │  │
│  │  4. Embedding Generation (sentence-transformers)                 │  │
│  │  5. Vector Search (FAISS)                                        │  │
│  │  6. Image Prioritization (for visual queries)                    │  │
│  │  7. LLM Generation (OpenAI/Ollama)                               │  │
│  │  8. Response Formatting (answer + citations + images)            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────┬─────────────────────────┬─────────────────────────────────┘
              │                         │
              ▼                         ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│    EMBEDDER             │   │    LLM PROVIDER         │
│ (src/rag/embeddings.py) │   │   (src/rag/llm.py)      │
│                         │   │                         │
│  - Model Loading        │   │  - OpenAI Client        │
│  - Text Normalization   │   │  - Ollama Client        │
│  - Batch Processing     │   │  - Prompt Building      │
│  - Vector Generation    │   │  - Response Parsing     │
│    (384-dim)            │   │  - Error Handling       │
└─────────────────────────┘   └─────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     VECTOR STORE                                         │
│                  (src/rag/vectorstore.py)                                │
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │ FAISS Index  │  │ BM25 Index   │  │  Metadata    │                 │
│  │ (vectors)    │  │ (keywords)   │  │  (JSONL)     │                 │
│  │ 57MB         │  │ 6MB          │  │  102MB       │                 │
│  │ IndexFlatIP  │  │ Tokenized    │  │  text+images │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
│                                                                           │
│  Operations:                                                             │
│  - search() - Pure vector or hybrid BM25+vector                         │
│  - add_documents() - Incremental addition                               │
│  - remove_document() - Delete and rebuild                               │
│  - needs_update() - Hash-based change detection                         │
└─────────────────────────────────────────────────────────────────────────┘
              ▲
              │
              │ (reads/writes)
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DISK STORAGE                                       │
│                     (data/index/)                                        │
│                                                                           │
│  📄 faiss.index        - Vector database (FAISS binary format)          │
│  📄 bm25.pkl           - BM25 tokenized corpus (pickle)                 │
│  📄 metadata.jsonl     - Chunk metadata with images (JSON Lines)        │
│  📄 doc_hashes.json    - File SHA256 hashes for change detection        │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION PIPELINE                           │
│                                                                           │
│  ┌───────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐            │
│  │   Auto    │──▶│  Ingest  │──▶│ Loader  │──▶│ Chunker  │            │
│  │  Indexer  │   │ Pipeline │   │         │   │          │            │
│  └───────────┘   └──────────┘   └─────────┘   └──────────┘            │
│       │              │                │              │                   │
│       │              │                │              ▼                   │
│       ▼              │                │         ┌──────────┐            │
│  ┌───────────┐      │                │         │ Embedder │            │
│  │ Watchdog  │      │                │         └──────────┘            │
│  │  Monitor  │      │                │              │                   │
│  │           │      │                ▼              ▼                   │
│  │ Detects:  │      │         ┌──────────────────────────┐            │
│  │ - Create  │      │         │  Format-Specific Loaders │            │
│  │ - Modify  │      │         ├──────────────────────────┤            │
│  │ - Delete  │      │         │ PDF   → pypdf + images   │            │
│  └───────────┘      │         │ DOCX  → python-docx      │            │
│       │              │         │ XLSX  → openpyxl+tables  │            │
│       │              │         │ HTML  → bs4+tables       │            │
│       │              │         │ CSV   → csv+tables       │            │
│       ▼              │         │ TXT   → encoding detect  │            │
│  ┌───────────┐      │         └──────────────────────────┘            │
│  │ Debounce  │      │                │                                  │
│  │  (2 sec)  │      │                ▼                                  │
│  └───────────┘      │         ┌──────────────┐                         │
│       │              │         │ Text + Images│                         │
│       │              │         └──────────────┘                         │
│       ▼              │                │                                  │
│  ┌───────────┐      │                ▼                                  │
│  │  Trigger  │──────┘         ┌──────────────┐                         │
│  │  Ingest   │                │  Chunking     │                         │
│  └───────────┘                │  (1000 tokens)│                         │
│                                │  (200 overlap)│                         │
│                                └──────────────┘                         │
│                                       │                                  │
│                                       ▼                                  │
│                                ┌──────────────┐                         │
│                                │ Vector Store │                         │
│                                │   Update     │                         │
│                                └──────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                    │
│                          (data/ folder)                                  │
│                                                                           │
│  📁 docs/                   - User documents (monitored)                │
│  📁 example-docs/           - Sample files (monitored)                  │
│  📁 index/                  - Generated index (excluded from watch)     │
│                                                                           │
│  Supported: PDF, DOCX, PPTX, XLSX, HTML, XML, EML, CSV, JSON, MD, TXT  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. FastAPI Server (`src/api/main.py`)

**Responsibilities:**
- HTTP endpoint management
- Request validation
- Startup initialization
- Auto-indexer lifecycle

**Key Functions:**
```python
@app.on_event("startup")
def _startup():
    # 1. Check if index exists
    # 2. Run initial indexing if missing
    # 3. Initialize ChatService
    # 4. Start AutoIndexer (file watcher)

@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Validate request
    # 2. Call chat_service.answer()
    # 3. Return ChatResponse
```

**Endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve web UI |
| `/chat` | POST | Answer questions |
| `/health` | GET | Health check |

---

### 2. Chat Service (`src/rag/chat.py`)

**Responsibilities:**
- Query processing pipeline
- Safety filtering
- Query reformulation
- Retrieval coordination
- LLM answer generation

**Pipeline Flow:**
```
User Question
    │
    ▼
┌─────────────────┐
│ Safety Filter   │ → Check for PII/PHI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Reform.   │ → LLM reformulates follow-ups
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visual Detect   │ → Check for diagram/chart keywords
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding       │ → Convert to 384-dim vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Search   │ → FAISS similarity search (pure semantic)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Priority  │ → Sort image chunks first (if visual query)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Generate    │ → GPT-4o-mini with context
└────────┬────────┘
         │
         ▼
    Response
```

**Key Methods:**
```python
async def answer(question: str, history: list):
    # 1. safety_filter.check_content_safety(question)
    # 2. search_query = _reformulate_with_context(question, history)
    # 3. qvec = embedder.embed_texts([search_query])
    # 4. retrieved = store.search(qvec, top_k=5)
    # 5. text = llm.complete(prompt)
    # 6. return {answer, retrieved}

async def _reformulate_with_context(question: str, history: list):
    # Uses LLM to understand conversation context
    # Handles pronouns and references
```

---

### 3. Vector Store (`src/rag/vectorstore.py`)

**Responsibilities:**
- FAISS index management
- BM25 index management (optional)
- Metadata storage
- Incremental updates
- Change detection

**Data Structures:**
```python
class FaissStore:
    index: faiss.IndexFlatIP           # Vector index (inner product)
    bm25: BM25Okapi                    # Keyword index
    meta: list[dict]                   # Chunk metadata
    doc_hashes: dict[str, str]         # filename -> SHA256 hash
    tokenized_corpus: list[list[str]]  # For BM25
```

**Key Operations:**

**Search (Pure Vector - Default):**
```python
def _vector_search(query_vec, top_k):
    # 1. FAISS similarity search
    # 2. Deduplicate by source (keep latest)
    # 3. Sort by score descending
    # 4. Return top_k results
```

**Search (Hybrid - Optional):**
```python
def _hybrid_search(query_vec, query_text, top_k, bm25_weight):
    # 1. Get vector search results
    # 2. Get BM25 search results
    # 3. Reciprocal Rank Fusion (RRF)
    #    score = (1-w) * vector_score + w * bm25_score
    # 4. Deduplicate and sort
    # 5. Return top_k results
```

**Incremental Update:**
```python
def add_documents(vectors, chunks, file_hash):
    # 1. Add vectors to FAISS index
    # 2. Append to metadata list
    # 3. Update tokenized corpus
    # 4. Rebuild BM25 (fast operation)
    # 5. Store file hash
```

**Delete Document:**
```python
def remove_document(source):
    # 1. Find indices to keep
    # 2. Extract vectors from old index
    # 3. Create new FAISS index with filtered vectors
    # 4. Update metadata
    # 5. Rebuild BM25
```

---

### 4. Document Loader (`src/rag/unstructured_loader.py`)

**Responsibilities:**
- Multi-format parsing
- Text extraction
- Image extraction (PDF)
- Table preservation
- Encoding detection

**Format Handlers:**

| Format | Library | Special Handling |
|--------|---------|------------------|
| PDF | pypdf | Extract images from /XObject, filter <50x50px or >1MB |
| DOCX | python-docx | Extract paragraphs, preserve styles |
| XLSX | openpyxl | Convert to markdown tables, filter empty rows |
| HTML | BeautifulSoup4 | Convert `<table>` to markdown, remove scripts |
| CSV | csv module | Convert to markdown tables, detect delimiter |
| JSON | json module | Pretty print or flatten |
| TXT/MD | built-in | Detect encoding (UTF-8/16/32) |

**Table Extraction Example:**
```python
# HTML/Excel/CSV → Markdown Table
Input:  <table><tr><th>Name</th><th>Age</th></tr>...</table>
Output: | Name | Age |
        | --- | --- |
        | John | 30 |
```

**Image Extraction (PDF):**
```python
def _extract_images(page):
    # 1. Access /Resources/XObject
    # 2. For each object:
    #    - Check if it's an image
    #    - Filter by size (50x50 < size < 1MB)
    #    - Extract bytes
    #    - Base64 encode
    #    - Store metadata (page, format, dimensions)
```

---

### 5. Embedder (`src/rag/embeddings.py`)

**Responsibilities:**
- Model loading (sentence-transformers)
- Text embedding generation
- Batch processing
- Vector normalization

**Implementation:**
```python
class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        # all-MiniLM-L6-v2: 384-dim, 80MB, ~50ms/query
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # 1. Batch processing (up to 32 at once)
        # 2. Model inference
        # 3. Normalize vectors (for cosine similarity)
        # 4. Return numpy array (N x 384)
```

**Normalization:**
- Vectors are L2-normalized for cosine similarity
- FAISS IndexFlatIP uses inner product (equivalent to cosine for normalized vectors)

---

### 6. LLM Provider (`src/rag/llm.py`)

**Responsibilities:**
- LLM client management
- Prompt construction
- API communication
- Response parsing

**Providers:**

**OpenAI:**
```python
class OpenAILLM:
    async def complete(self, prompt: str) -> str:
        # 1. Construct messages [system, user]
        # 2. Call OpenAI API
        # 3. Parse response
        # 4. Handle rate limits/errors
```

**Ollama (Local):**
```python
class OllamaLLM:
    async def complete(self, prompt: str) -> str:
        # 1. HTTP request to localhost:11434
        # 2. Stream or batch response
        # 3. Parse JSON
```

---

### 7. Auto Indexer (`src/rag/auto_indexer.py`)

**Responsibilities:**
- File system monitoring
- Change detection
- Automatic re-indexing
- Event debouncing

**Implementation:**
```python
class AutoIndexer:
    def __init__(self, docs_dir, store, embedder, index_dir):
        self.observer = Observer()  # watchdog
        self.handler = FileChangeHandler()
    
    def start():
        # Start watchdog observer
        # Watch docs_dir recursively
        # Exclude index_dir
    
    def on_created(event):
        # Wait 2 seconds (debounce)
        # Load document
        # Generate embeddings
        # store.add_documents()
    
    def on_modified(event):
        # Remove old version
        # Add new version
    
    def on_deleted(event):
        # store.remove_document()
```

**Debouncing:**
- Waits 2 seconds after file write completes
- Prevents multiple triggers for same file
- Uses threading.Timer

---

### 8. Chunking (`src/rag/chunking.py`)

**Responsibilities:**
- Text splitting
- Overlap management
- Chunk metadata

**Strategy:**
```python
CHUNK_SIZE = 1000   # tokens
CHUNK_OVERLAP = 200 # tokens

def chunk_text(text, source, images):
    # 1. Tokenize with tiktoken
    # 2. Split into chunks of CHUNK_SIZE
    # 3. Add CHUNK_OVERLAP between chunks
    # 4. Associate images with first chunk
    # 5. Return list of Chunk objects
```

**Chunk Object:**
```python
@dataclass
class Chunk:
    source: str          # Filename
    chunk_id: int        # 0, 1, 2, ...
    text: str            # Chunk text
    timestamp: float     # Unix timestamp
    images: list[dict]   # [{"page": 1, "format": "PNG", "data": "base64..."}]
```

---

### 9. Configuration (`src/rag/config.py`)

**Responsibilities:**
- Environment variable loading
- Default values
- Path resolution

**Settings:**
```python
class Settings:
    docs_dir: str = "./data"
    index_dir: str = "./data/index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_model: str = "gpt-4o-mini"
    top_k: int = 5
    
settings = Settings()
```

---

## Data Flow

### 1. Document Ingestion Flow

```
┌─────────────┐
│  Add File   │
│ to data/    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Watchdog   │───┐
│  Detects    │   │ (debounce 2s)
└──────┬──────┘   │
       │          │
       ▼          │
┌─────────────┐  │
│  Compute    │  │
│  SHA256     │  │
│  Hash       │  │
└──────┬──────┘  │
       │          │
       ▼          │
┌─────────────┐  │
│  Check      │  │
│  if Changed │  │
└──────┬──────┘  │
       │          │
       ▼          │
┌─────────────┐  │
│  Load       │  │
│  Document   │  │
│  (format-   │  │
│   specific) │  │
└──────┬──────┘  │
       │          │
       ▼          │
┌─────────────┐  │
│  Extract    │  │
│  - Text     │  │
│  - Images   │  │
│  - Tables   │  │
└──────┬──────┘  │
       │          │
       ▼          │
┌─────────────┐  │
│  Chunk      │  │
│  (1000 tok, │  │
│   200 over) │  │
└──────┬──────┘  │
       │          │
       ▼          │
┌─────────────┐  │
│  Generate   │  │
│  Embeddings │  │
│  (384-dim)  │  │
└──────┬──────┘  │
       │          │
       ▼          │
┌─────────────┐  │
│  Add to     │  │
│  FAISS      │  │
│  Index      │  │
└──────┬──────┘  │
       │          │
       ▼          │
┌─────────────┐  │
│  Update     │  │
│  Metadata   │  │
│  & Hashes   │  │
└──────┬──────┘  │
       │          │
       ▼          │
┌─────────────┐  │
│  Save to    │◀─┘
│  Disk       │
└─────────────┘
```

### 2. Query Processing Flow

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Safety      │──→ Reject if PII/PHI detected
│ Filter      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Check for   │
│ Follow-up   │
│ (history    │
│  length>0)  │
└──────┬──────┘
       │
       ├─── Yes ──→ ┌──────────────┐
       │            │ Reformulate  │
       │            │ with LLM     │
       │            └──────┬───────┘
       │                   │
       ▼                   ▼
       └──────────┬────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Visual Query?    │
       │ (diagram/chart)  │
       └──────┬───────────┘
                  │
       ┌──────────┴──────────┐
       │ Yes          No     │
       │ top_k*2      top_k  │
       └──────────┬──────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Generate         │
       │ Embedding        │
       │ (384-dim vector) │
       └──────┬───────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Vector Search    │
       │ (FAISS)          │
       │ Pure Semantic    │
       └──────┬───────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Deduplicate      │
       │ by Source        │
       │ (keep latest)    │
       └──────┬───────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Visual Query?    │
       └──────┬───────────┘
                  │
       ┌──────────┴──────────┐
       │ Yes          No     │
       │ Prioritize   Keep   │
       │ img chunks   order  │
       └──────────┬──────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Build Context    │
       │ from Retrieved   │
       │ Chunks           │
       └──────┬───────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Build Prompt     │
       │ - System         │
       │ - Context        │
       │ - History        │
       │ - Question       │
       └──────┬───────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ LLM Generate     │
       │ (GPT-4o-mini)    │
       └──────┬───────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Format Response  │
       │ - Answer         │
       │ - Citations      │
       │ - Images         │
       └──────┬───────────┘
                  │
                  ▼
       ┌──────────────────┐
       │ Return to User   │
       └──────────────────┘
```

---

## Sequence Diagrams

### 1. Initial Startup Sequence

```
User                FastAPI             ChatService         VectorStore         FileSystem
 │                    │                     │                   │                   │
 │──uvicorn start────▶│                     │                   │                   │
 │                    │                     │                   │                   │
 │                    │──check index────────┼──────────────────▶│                   │
 │                    │                     │                   │                   │
 │                    │                     │                   │◀──exists?─────────│
 │                    │                     │                   │                   │
 │                    │◀─index missing──────┼───────────────────│                   │
 │                    │                     │                   │                   │
 │                    │──run_ingest()───────▶                   │                   │
 │                    │                     │                   │                   │
 │                    │                     │──scan docs────────┼──────────────────▶│
 │                    │                     │                   │                   │
 │                    │                     │◀─file list────────┼───────────────────│
 │                    │                     │                   │                   │
 │                    │                     │──load+embed───────▶                   │
 │                    │                     │                   │                   │
 │                    │                     │──build index──────▶                   │
 │                    │                     │                   │                   │
 │                    │                     │──save─────────────┼──────────────────▶│
 │                    │                     │                   │                   │
 │                    │◀─complete───────────│                   │                   │
 │                    │                     │                   │                   │
 │                    │──init ChatService──▶│                   │                   │
 │                    │                     │                   │                   │
 │                    │                     │──load index───────▶                   │
 │                    │                     │                   │                   │
 │                    │◀─service ready──────│                   │                   │
 │                    │                     │                   │                   │
 │                    │──start watcher──────▶                   │                   │
 │                    │                     │                   │                   │
 │◀─server ready──────│                     │                   │                   │
```

### 2. Chat Query Sequence

```
User         FastAPI      ChatService    SafetyFilter   Embedder    VectorStore    LLM
 │              │             │              │             │            │            │
 │──/chat POST─▶│             │              │             │            │            │
 │              │             │              │             │            │            │
 │              │──answer()──▶│              │             │            │            │
 │              │             │              │             │            │            │
 │              │             │──check_safety▶│            │            │            │
 │              │             │              │             │            │            │
 │              │             │◀─safe────────│             │            │            │
 │              │             │              │             │            │            │
 │              │             │──reformulate (if follow-up)──────────────────────────▶│
 │              │             │              │             │            │            │
 │              │             │◀─reformulated query────────────────────────────────────│
 │              │             │              │             │            │            │
 │              │             │──embed───────┼────────────▶│            │            │
 │              │             │              │             │            │            │
 │              │             │◀─vector──────┼─────────────│            │            │
 │              │             │              │             │            │            │
 │              │             │──search──────┼─────────────┼───────────▶│            │
 │              │             │              │             │            │            │
 │              │             │◀─top_k chunks┼─────────────┼────────────│            │
 │              │             │              │             │            │            │
 │              │             │──build prompt┼─────────────┼────────────┼───────────▶│
 │              │             │              │             │            │            │
 │              │             │◀─answer──────┼─────────────┼────────────┼────────────│
 │              │             │              │             │            │            │
 │              │◀─response───│              │             │            │            │
 │              │             │              │             │            │            │
 │◀─JSON resp───│             │              │             │            │            │
```

### 3. File Change Detection Sequence

```
FileSystem     AutoIndexer    Ingest      Loader     Embedder    VectorStore
    │               │            │           │           │            │
    │──file added──▶│            │           │           │            │
    │               │            │           │           │            │
    │               │──debounce──│           │           │            │
    │               │  (2 sec)   │           │           │            │
    │               │            │           │           │            │
    │               │──hash file─▶│          │           │            │
    │               │            │           │           │            │
    │               │            │──check────┼───────────┼───────────▶│
    │               │            │  exists?  │           │            │
    │               │            │           │           │            │
    │               │            │◀─new file─┼───────────┼────────────│
    │               │            │           │           │            │
    │               │            │──load─────▶│          │            │
    │               │            │           │           │            │
    │               │            │◀─text+img──│          │            │
    │               │            │           │           │            │
    │               │            │──chunk────▶│          │            │
    │               │            │           │           │            │
    │               │            │──embed────┼──────────▶│            │
    │               │            │           │           │            │
    │               │            │◀─vectors──┼───────────│            │
    │               │            │           │           │            │
    │               │            │──add docs─┼───────────┼───────────▶│
    │               │            │           │           │            │
    │               │            │──save─────┼───────────┼───────────▶│
    │               │            │           │           │            │
    │               │◀─complete──│           │           │            │
```

---

## Storage Structure

### Directory Layout

```
chat-with-docs/
├── data/
│   ├── docs/                    # User documents (watched)
│   │   ├── report.pdf
│   │   ├── spreadsheet.xlsx
│   │   └── ...
│   │
│   ├── example-docs/            # Sample files (watched)
│   │   └── ...
│   │
│   └── index/                   # Generated indices (excluded from watch)
│       ├── faiss.index          # FAISS vector database
│       ├── bm25.pkl             # BM25 tokenized corpus
│       ├── metadata.jsonl       # Chunk metadata with images
│       └── doc_hashes.json      # File change tracking
│
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   │
│   └── rag/
│       ├── __init__.py
│       ├── auto_indexer.py      # File watcher
│       ├── chat.py              # Query processing
│       ├── chunking.py          # Text splitting
│       ├── config.py            # Settings
│       ├── embeddings.py        # Embedding generation
│       ├── ingest.py            # Indexing pipeline
│       ├── llm.py               # LLM providers
│       ├── prompts.py           # System prompts
│       ├── unstructured_loader.py  # Document loaders
│       └── vectorstore.py       # FAISS + BM25 management
│
├── .env                         # Configuration
├── pyproject.toml               # Dependencies
└── README.md                    # Documentation
```

### Index File Formats

**1. faiss.index (Binary)**
```
Format: FAISS IndexFlatIP binary format
Size: 57MB for 38,863 vectors (384 dimensions)
Structure:
  - Header (index type, dimension)
  - Vector data (float32, L2-normalized)
  - Optimized for inner product search
```

**2. bm25.pkl (Pickle)**
```python
{
    "tokenized_corpus": [
        ["word1", "word2", ...],  # Document 0
        ["word3", "word4", ...],  # Document 1
        ...
    ]
}
# BM25Okapi object reconstructed from this
```

**3. metadata.jsonl (JSON Lines)**
```json
{"id": 0, "source": "doc.pdf", "chunk_id": 0, "text": "...", "timestamp": 1707123456.789, "images": [...]}
{"id": 1, "source": "doc.pdf", "chunk_id": 1, "text": "...", "timestamp": 1707123456.789, "images": []}
...
```

**4. doc_hashes.json (JSON)**
```json
{
  "document1.pdf": "a1b2c3d4e5f6...",
  "spreadsheet.xlsx": "f6e5d4c3b2a1...",
  ...
}
```

### Metadata Schema

**Chunk Metadata:**
```json
{
  "id": 42,
  "source": "building-blocks-of-rag.pdf",
  "chunk_id": 15,
  "text": "Retrieval-Augmented Generation (RAG) is...",
  "timestamp": 1707123456.789,
  "images": [
    {
      "page": 3,
      "format": "PNG",
      "width": 800,
      "height": 600,
      "data": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

**Image Metadata:**
```json
{
  "page": 3,
  "format": "PNG",
  "width": 800,
  "height": 600,
  "data": "base64_encoded_image_bytes"
}
```

---

## Key Algorithms

### 1. Reciprocal Rank Fusion (RRF)

**Purpose:** Combine BM25 and vector search scores

```python
def reciprocal_rank_fusion(vector_ranks, bm25_ranks, k=60, weight=0.15):
    """
    RRF Score = (1-w) * (1/(k + vector_rank)) + w * (1/(k + bm25_rank))
    
    Args:
        vector_ranks: {doc_id: rank} from vector search
        bm25_ranks: {doc_id: rank} from BM25 search
        k: RRF constant (default 60)
        weight: BM25 weight (0-1, default 0.15)
    
    Returns:
        Sorted list of (doc_id, score) tuples
    """
    all_docs = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    scores = {}
    
    for doc_id in all_docs:
        v_score = 1 / (k + vector_ranks.get(doc_id, 1000)) if doc_id in vector_ranks else 0
        b_score = 1 / (k + bm25_ranks.get(doc_id, 1000)) if doc_id in bm25_ranks else 0
        scores[doc_id] = (1 - weight) * v_score + weight * b_score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 2. Incremental Index Update

**Purpose:** Only re-index changed files

```python
def incremental_update(new_file_path, old_hash, new_hash):
    """
    1. Compare hashes
    2. If changed:
       a. Remove old vectors (rebuild FAISS)
       b. Load new document
       c. Generate embeddings
       d. Add to FAISS
       e. Update metadata
       f. Save index
    """
    if old_hash != new_hash:
        # File changed
        store.remove_document(filename)
        text, images = loader.load_document(new_file_path)
        chunks = chunk_text(text, filename, images)
        vectors = embedder.embed_texts([c.text for c in chunks])
        store.add_documents(vectors, chunks, new_hash)
        store.save()
```

### 3. Query Reformulation

**Purpose:** Handle follow-up questions with context

```python
async def reformulate_with_context(question, history):
    """
    Use LLM to reformulate question based on conversation history
    
    Example:
    History: 
      User: "What is RAG?"
      Assistant: "RAG is Retrieval-Augmented Generation..."
    
    Question: "Show me the architecture diagram"
    
    Reformulated: "Show me the RAG architecture diagram"
    """
    prompt = f"""
    Given conversation history, reformulate the question to be standalone.
    
    History: {history}
    Question: {question}
    
    Reformulated question:
    """
    
    return await llm.complete(prompt)
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Embedding generation | O(n) | n = text length, ~50ms per query |
| FAISS search | O(log n) | n = index size, <100ms for 40K vectors |
| BM25 search | O(n) | n = corpus size, ~20ms for 40K docs |
| LLM generation | O(1) | Network-bound, ~1-2s |
| Index build | O(n·d) | n = docs, d = avg doc size |
| Incremental add | O(k) | k = new doc chunks, ~2-5s |

### Space Complexity

| Component | Size | Scaling |
|-----------|------|---------|
| FAISS index | 57MB | O(n·d), d=384 dims |
| BM25 corpus | 6MB | O(n·v), v=vocab size |
| Metadata | 102MB | O(n·t+i), t=text, i=images |
| Total | 165MB | For 38,863 chunks |

---

## Configuration & Tuning

### Key Parameters

```python
# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunking
CHUNK_SIZE = 1000   # tokens
CHUNK_OVERLAP = 200 # tokens

# Retrieval
TOP_K = 5
VISUAL_QUERY_MULTIPLIER = 2  # top_k * 2 for diagram queries

# Hybrid Search (if enabled)
BM25_WEIGHT = 0.15  # 15% BM25, 85% vector
RRF_K = 60          # RRF constant

# LLM
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0   # Deterministic

# Auto-indexer
DEBOUNCE_SECONDS = 2
```

### Tuning Guidelines

**For better semantic search:**
- Increase embedding dimension (use text-embedding-3-large)
- Reduce chunk size for granular retrieval
- Increase TOP_K for more context

**For better keyword matching:**
- Enable hybrid search (set `hybrid=True`)
- Increase BM25_WEIGHT (0.3-0.5)
- Use better tokenization (stemming)

**For faster indexing:**
- Increase CHUNK_SIZE (reduce total chunks)
- Use smaller embedding model
- Disable BM25 index

**For better answers:**
- Increase TOP_K (more context)
- Use GPT-4 instead of GPT-4o-mini
- Improve chunking (structure-aware)

---

## Error Handling

### Startup Errors

```python
try:
    # Check index exists
    if not index_exists():
        run_ingest(reset=True)
    
    chat_service = ChatService()
    auto_indexer = AutoIndexer()
    auto_indexer.start()
    
except Exception as e:
    logger.error(f"Startup failed: {e}")
    raise  # FastAPI will exit
```

### Query Errors

```python
try:
    # Safety check
    is_safe, reason = safety_filter.check(question)
    if not is_safe:
        raise RuntimeError(reason)
    
    # Search
    retrieved = store.search(qvec, top_k)
    
    if not retrieved:
        return {"answer": "No relevant documents found"}
    
    # LLM generation
    answer = await llm.complete(prompt)
    
except RuntimeError as e:
    # User-facing error
    raise HTTPException(status_code=400, detail=str(e))

except Exception as e:
    # Internal error
    logger.error(f"Query failed: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Internal error")
```

### Indexing Errors

```python
def process_file(path):
    try:
        text, images = loader.load_document(path)
        
        if not text:
            logger.warning(f"Empty document: {path}")
            return
        
        chunks = chunk_text(text, path.name, images)
        vectors = embedder.embed_texts([c.text for c in chunks])
        store.add_documents(vectors, chunks, file_hash)
        
    except Exception as e:
        logger.error(f"Failed to process {path}: {e}")
        # Continue with next file (don't crash)
```

---

## Testing Strategy

### Unit Tests

```python
# Test individual components
test_embedder()          # Embedding generation
test_vectorstore()       # FAISS operations
test_loader()            # Document loading
test_chunking()          # Text splitting
test_reformulation()     # Query processing
```

### Integration Tests

```python
# Test component interactions
test_ingest_pipeline()   # Load → Chunk → Embed → Index
test_search_pipeline()   # Query → Embed → Search → Rank
test_chat_pipeline()     # Question → Search → LLM → Answer
```

### End-to-End Tests

```python
# Test full system
test_add_document()      # Add file → Auto-index → Query
test_update_document()   # Modify file → Re-index → Query
test_delete_document()   # Delete file → Remove from index
test_conversation()      # Multi-turn Q&A with context
```

---

## Deployment Considerations

### Production Checklist

- [ ] Set strong `OPENAI_API_KEY`
- [ ] Configure logging (file + rotation)
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Enable HTTPS (reverse proxy)
- [ ] Set rate limits
- [ ] Configure CORS appropriately
- [ ] Set up backup for `data/index/`
- [ ] Monitor disk space (index grows)
- [ ] Set resource limits (memory, CPU)
- [ ] Configure error alerting

### Scaling

**Vertical Scaling (Single Server):**
- Increase RAM for larger indices
- Use GPU for faster embeddings
- SSD for faster I/O

**Horizontal Scaling (Multiple Servers):**
- Separate indexing and query servers
- Use shared storage (NFS/S3) for index
- Load balance query API
- Distributed FAISS (faiss-gpu, IVF index)

---

## Maintenance

### Regular Tasks

**Daily:**
- Monitor index size growth
- Check error logs
- Verify auto-indexer running

**Weekly:**
- Review slow queries
- Check disk space
- Update dependencies

**Monthly:**
- Optimize index (rebuild if fragmented)
- Evaluate retrieval quality
- Update embedding model if needed

### Troubleshooting

**Problem: Slow queries**
- Check index size (>100K chunks?)
- Profile with logging
- Consider IVF index for large scale

**Problem: Poor retrieval**
- Check chunk size (too large?)
- Verify embedding quality
- Test with different queries
- Try enabling hybrid search

**Problem: Out of memory**
- Reduce batch size in embedder
- Increase swap space
- Use smaller embedding model

---

## Future Enhancements

### Planned Improvements

1. **Multi-modal Embeddings (CLIP)**
   - Joint text-image search
   - Better visual query matching

2. **Query Expansion**
   - Automatic synonym generation
   - Related term injection

3. **Reranking**
   - Cross-encoder for top-k refinement
   - Improves precision

4. **Streaming Responses**
   - Server-Sent Events (SSE)
   - Real-time answer generation

5. **Evaluation Suite**
   - Automated quality metrics
   - Regression testing

---

## References

### Papers & Resources

- **RAG**: Lewis et al. "Retrieval-Augmented Generation" (2020)
- **BM25**: Robertson & Zaragoza "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
- **RRF**: Cormack et al. "Reciprocal Rank Fusion" (2009)
- **FAISS**: Johnson et al. "Billion-scale similarity search with GPUs" (2017)
- **Sentence Transformers**: Reimers & Gurevych "Sentence-BERT" (2019)

### Libraries

- FastAPI: https://fastapi.tiangolo.com/
- FAISS: https://github.com/facebookresearch/faiss
- Sentence Transformers: https://www.sbert.net/
- rank-bm25: https://github.com/dorianbrown/rank_bm25
- watchdog: https://github.com/gorakhargosh/watchdog

---

**Last Updated:** 2026-02-05
**Version:** 0.2.0
**Author:** System Design Documentation
