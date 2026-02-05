"""
Auto-watch file system for document changes and incrementally update the index.
"""
from __future__ import annotations
import hashlib
import logging
import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from .config import settings
from .unstructured_loader import get_loader, SUPPORTED_EXTENSIONS
from .chunking import chunk_text
from .embeddings import Embedder
from .vectorstore import FaissStore

logger = logging.getLogger(__name__)

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {e}")
        return ""

class DocumentWatcher(FileSystemEventHandler):
    """Watch for file changes and update index incrementally"""
    
    def __init__(self, docs_dir: str, store: FaissStore, embedder: Embedder, index_dir: str):
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir).resolve()
        self.store = store
        self.embedder = embedder
        self.loader = get_loader()
        self.processing = set()  # Track files being processed
        
    def should_process(self, path: Path) -> bool:
        """Check if file should be processed"""
        if not path.is_file():
            return False
        # Skip files in index directory
        if self.index_dir in path.resolve().parents or path.resolve() == self.index_dir:
            return False
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        if 'dont_want' in path.name.lower() or path.name.startswith('.'):
            return False
        if str(path) in self.processing:
            return False
        return True
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation"""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        if not self.should_process(path):
            return
        
        logger.info(f"📄 Detected new file: {path.name}")
        # Process in background thread to avoid blocking
        thread = threading.Thread(target=self.process_new_file_sync, args=(path,))
        thread.daemon = True
        thread.start()
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification"""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        if not self.should_process(path):
            return
        
        logger.info(f"📝 Detected modified file: {path.name}")
        thread = threading.Thread(target=self.process_modified_file_sync, args=(path,))
        thread.daemon = True
        thread.start()
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion"""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        
        logger.info(f"🗑️ Detected deleted file: {path.name}")
        thread = threading.Thread(target=self.process_deleted_file_sync, args=(path,))
        thread.daemon = True
        thread.start()
    
    def process_new_file_sync(self, path: Path):
        """Process newly created file"""
        if str(path) in self.processing:
            return
        
        self.processing.add(str(path))
        
        try:
            # Small delay to ensure file is fully written
            import time
            time.sleep(0.5)
            
            if not path.exists():
                return
            
            # Compute file hash
            file_hash = compute_file_hash(path)
            if not file_hash:
                return
            
            # Load document
            text, images = self.loader.load_document(path)
            if not text or len(text.strip()) < 50:
                logger.warning(f"Skipped {path.name}: empty or too short")
                return
            
            # Create chunks
            chunks = chunk_text(text, source=str(path.name), images=images)
            if not chunks:
                return
            
            # Create embeddings
            vectors = self.embedder.embed_texts([c.text for c in chunks])
            
            # Add to store
            self.store.add_documents(vectors, chunks, file_hash)
            self.store.save()
            
            import datetime
            timestamp = chunks[0].timestamp if chunks else time.time()
            dt_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"✅ Added {path.name}: {len(chunks)} chunks (v{timestamp:.0f} @ {dt_str})")
            
        except Exception as e:
            logger.error(f"Error processing {path.name}: {e}")
        finally:
            self.processing.discard(str(path))
    
    def process_modified_file_sync(self, path: Path):
        """Process modified file"""
        if str(path) in self.processing:
            return
        
        self.processing.add(str(path))
        
        try:
            # Small delay to ensure file is fully written
            import time
            time.sleep(0.5)
            
            if not path.exists():
                return
            
            # Compute new hash
            file_hash = compute_file_hash(path)
            if not file_hash:
                return
            
            # Check if actually changed
            if not self.store.needs_update(path.name, file_hash):
                return
            
            # Remove old version
            self.store.remove_document(path.name)
            
            # Load and process new version
            text, images = self.loader.load_document(path)
            if not text or len(text.strip()) < 50:
                logger.warning(f"Skipped {path.name}: empty or too short")
                self.store.save()
                return
            
            # Create chunks
            chunks = chunk_text(text, source=str(path.name), images=images)
            if not chunks:
                self.store.save()
                return
            
            # Create embeddings
            vectors = self.embedder.embed_texts([c.text for c in chunks])
            
            # Add to store
            self.store.add_documents(vectors, chunks, file_hash)
            self.store.save()
            
            import datetime
            timestamp = chunks[0].timestamp if chunks else time.time()
            dt_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"♻️ Updated {path.name}: {len(chunks)} chunks (v{timestamp:.0f} @ {dt_str})")
            
        except Exception as e:
            logger.error(f"Error updating {path.name}: {e}")
        finally:
            self.processing.discard(str(path))
    
    def process_deleted_file_sync(self, path: Path):
        """Process deleted file"""
        try:
            # Remove from store
            removed = self.store.remove_document(path.name)
            
            if removed:
                self.store.save()
                logger.info(f"✅ Removed {path.name} from index and rebuilt")
            else:
                logger.info(f"⚠️  {path.name} not found in index")
            
        except Exception as e:
            logger.error(f"Error removing {path.name}: {e}")


class AutoIndexer:
    """Manage automatic indexing with file watching"""
    
    def __init__(self, docs_dir: str, store: FaissStore, embedder: Embedder, index_dir: str):
        self.docs_dir = docs_dir
        self.observer = Observer()
        self.handler = DocumentWatcher(docs_dir, store, embedder, index_dir)
        self.running = False
    
    def start(self):
        """Start watching for file changes"""
        if self.running:
            return
        
        self.observer.schedule(self.handler, str(self.docs_dir), recursive=True)
        self.observer.start()
        self.running = True
        logger.info(f"🔍 Auto-indexer started watching: {self.docs_dir}")
    
    def stop(self):
        """Stop watching"""
        if not self.running:
            return
        
        self.observer.stop()
        self.observer.join()
        self.running = False
        logger.info("🛑 Auto-indexer stopped")
