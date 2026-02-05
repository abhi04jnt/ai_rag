from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from rag.schemas import ChatRequest, ChatResponse
from rag.chat import ChatService
from rag.auto_indexer import AutoIndexer
from rag.config import settings
from rag.ingest import run_ingest
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chat With Your Docs", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_service: ChatService | None = None
auto_indexer: AutoIndexer | None = None

@app.on_event("startup")
def _startup():
    global chat_service, auto_indexer
    
    try:
        # Check if index exists, if not, run initial indexing ONCE
        index_path = Path(settings.index_dir) / "faiss.index"
        meta_path = Path(settings.index_dir) / "metadata.jsonl"
        
        if not index_path.exists() or not meta_path.exists():
            logger.info("📦 No index found. Running initial indexing (one-time only)...")
            logger.info(f"📂 Scanning all documents in: {settings.docs_dir}")
            run_ingest(reset=True)
            logger.info("✅ Initial indexing complete!")
        else:
            logger.info("✅ Index exists - skipping re-indexing on startup")
        
        chat_service = ChatService()
        logger.info("✅ Chat service initialized")
        
        # Start auto-indexer - it will handle ALL future changes silently in background
        auto_indexer = AutoIndexer(
            settings.docs_dir,
            chat_service.store,
            chat_service.embedder,
            settings.index_dir
        )
        auto_indexer.start()
        logger.info(f"🔍 Auto-indexer active: watching {settings.docs_dir} for changes")
        logger.info("   → File changes will be indexed automatically in background")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
def _shutdown():
    global auto_indexer
    if auto_indexer:
        auto_indexer.stop()
        logger.info("🛑 Auto-indexer stopped")

@app.get("/")
def read_root():
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "index.html")

@app.get("/health")
def health():
    return {
        "ok": True,
        "auto_indexer": auto_indexer.running if auto_indexer else False,
        "docs_dir": settings.docs_dir
    }

@app.get("/documents")
def list_documents():
    """List all indexed documents with version information"""
    assert chat_service is not None
    
    # Get unique sources
    sources = set(m["source"] for m in chat_service.store.meta)
    
    documents = []
    for source in sorted(sources):
        info = chat_service.store.get_document_info(source)
        if info:
            documents.append(info)
    
    return {
        "total_documents": len(documents),
        "total_chunks": len(chat_service.store.meta),
        "documents": documents
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    assert chat_service is not None
    try:
        result = await chat_service.answer(
            question=req.question,
            history=[m.model_dump() for m in req.history],
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
