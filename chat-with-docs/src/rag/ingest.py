from __future__ import annotations
from .config import settings
from .unstructured_loader import get_loader
from .chunking import chunk_text, Chunk
from .embeddings import Embedder
from .vectorstore import FaissStore
import hashlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with filepath.open('rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def run_ingest(reset: bool = False, docs_dir: str = None, incremental: bool = True) -> None:
    """
    Ingest documents and build vector index with incremental updates.
    
    Args:
        reset: If True, reset the index before building
        docs_dir: Directory containing documents (defaults to settings.docs_dir)
        incremental: If True, only process changed/new files (default: True)
    """
    docs_directory = docs_dir or settings.docs_dir
    
    store = FaissStore(settings.index_dir)
    
    if reset:
        logger.info("🔄 Resetting index (full rebuild)...")
        store.reset()
        incremental = False
    else:
        # Try to load existing index for incremental updates
        try:
            store.load()
            logger.info(f"📚 Loaded existing index: {len(store.meta)} chunks from {len(set(m['source'] for m in store.meta))} documents")
        except FileNotFoundError:
            logger.info("📚 No existing index found - creating new index")
            incremental = False

    loader = get_loader()
    embedder = Embedder(settings.embedding_model)
    
    logger.info(f"📂 Scanning documents recursively in: {docs_directory}")
    logger.info("   (excluding index/ folder)")
    
    # Find all documents
    docs_path = Path(docs_directory)
    index_path = Path(settings.index_dir).resolve()
    
    all_document_paths = {}  # path -> file_hash
    for ext in loader.get_supported_extensions():
        for path in docs_path.rglob(f"*{ext}"):
            # Skip files in index directory
            if index_path in path.resolve().parents or path.resolve() == index_path:
                continue
            # Skip hidden files and unwanted files
            if path.name.startswith('.') or 'dont_want' in path.name.lower():
                continue
            
            file_hash = compute_file_hash(path)
            all_document_paths[path] = file_hash
    
    logger.info(f"   Found {len(all_document_paths)} documents across all folders")
    
    if incremental:
        # Determine what needs to be updated
        existing_sources = {m['source'] for m in store.meta}
        current_sources = {str(p.name) for p in all_document_paths.keys()}
        
        # Files to add/update
        files_to_process = []
        for path, file_hash in all_document_paths.items():
            source = str(path.name)
            if source not in existing_sources or store.needs_update(source, file_hash):
                files_to_process.append((path, file_hash))
        
        # Files to remove (deleted from disk)
        files_to_remove = existing_sources - current_sources
        
        logger.info(f"\n📊 Incremental Update:")
        logger.info(f"   ✓ {len(existing_sources)} existing documents")
        logger.info(f"   + {len(files_to_process)} to add/update")
        logger.info(f"   - {len(files_to_remove)} to remove")
        
        # Remove deleted files
        for source in files_to_remove:
            store.remove_document(source)
        
        # Process new/changed files
        for path, file_hash in files_to_process:
            try:
                logger.info(f"  📄 Processing: {path.name} ({path.suffix})")
                
                # Remove old version if exists
                source = str(path.name)
                if source in existing_sources:
                    logger.info("    🔄 Updating existing document")
                    store.remove_document(source)
                
                text, images = loader.load_document(path)
                
                if not text or len(text.strip()) < 50:
                    logger.warning(f"    ⚠️  Skipped (empty or too short): {path.name}")
                    continue
                
                chunks = chunk_text(text, source=source, images=images)
                
                # Create embeddings for this document
                vectors = embedder.embed_texts([c.text for c in chunks])
                
                # Add to index incrementally
                store.add_documents(vectors, chunks, file_hash)
                
                img_info = f", {len(images)} images" if images else ""
                logger.info(f"    ✓ Added {len(chunks)} chunks ({len(text)} chars{img_info})")
                
            except Exception as e:
                logger.error(f"    ❌ Error processing {path.name}: {e}")
                continue
        
        # Save updated index
        if files_to_process or files_to_remove:
            logger.info(f"\n💾 Saving updated index to {settings.index_dir}...")
            store.save()
            logger.info("✅ Incremental update complete!")
            logger.info(f"   - {len(store.meta)} total chunks")
            logger.info(f"   - {len(set(m['source'] for m in store.meta))} total documents")
        else:
            logger.info("\n✅ No changes detected - index is up to date!")
    
    else:
        # Full rebuild
        all_chunks: list[Chunk] = []
        doc_count = 0
        
        for path, file_hash in all_document_paths.items():
            try:
                logger.info(f"  📄 Processing: {path.name} ({path.suffix})")
                text, images = loader.load_document(path)
                
                if not text or len(text.strip()) < 50:
                    logger.warning(f"    ⚠️  Skipped (empty or too short): {path.name}")
                    continue
                
                chunks = chunk_text(text, source=str(path.name), images=images)
                all_chunks.extend(chunks)
                doc_count += 1
                
                # Store file hash
                store.doc_hashes[str(path.name)] = file_hash
                
                img_info = f", {len(images)} images" if images else ""
                logger.info(f"    ✓ Extracted {len(chunks)} chunks ({len(text)} chars{img_info})")
                
            except Exception as e:
                logger.error(f"    ❌ Error processing {path.name}: {e}")
                continue

        if not all_chunks:
            raise RuntimeError(f"No documents found or processed in {docs_directory}")

        logger.info(f"\n📊 Total: {doc_count} documents, {len(all_chunks)} chunks")
        logger.info("🔄 Creating embeddings...")
        
        vectors = embedder.embed_texts([c.text for c in all_chunks])

        logger.info(f"💾 Saving index to {settings.index_dir}...")
        store.build(vectors, all_chunks)
        store.save()
        
        logger.info("✅ Indexing complete!")
        logger.info(f"   - {len(all_chunks)} chunks indexed")
        logger.info(f"   - From {doc_count} documents")
        logger.info(f"   - Saved to {settings.index_dir}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents and build vector index")
    parser.add_argument("--reset", action="store_true", help="Reset index and rebuild from scratch")
    parser.add_argument("--full", action="store_true", help="Full rebuild (disable incremental)")
    parser.add_argument("docs_dir", nargs="?", default=None, help="Directory containing documents")
    
    args = parser.parse_args()
    
    run_ingest(reset=args.reset, docs_dir=args.docs_dir, incremental=not args.full)
