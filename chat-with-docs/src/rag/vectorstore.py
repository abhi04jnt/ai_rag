from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import logging
import numpy as np
import faiss
import hashlib
import pickle
from typing import Optional
from rank_bm25 import BM25Okapi

from .chunking import Chunk

logger = logging.getLogger(__name__)

class FaissStore:
    """
    Stores:
      - FAISS index (vectors) for semantic search
      - BM25 index for keyword search
      - metadata.jsonl with {id, source, chunk_id, text, file_hash}
    """
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.bm25_path = self.index_dir / "bm25.pkl"
        self.meta_path = self.index_dir / "metadata.jsonl"
        self.doc_hashes_path = self.index_dir / "doc_hashes.json"

        self.index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []
        self.meta: list[dict] = []
        self.doc_hashes: dict[str, str] = {}  # filename -> hash

    def reset(self):
        if self.index_path.exists():
            self.index_path.unlink()
        if self.bm25_path.exists():
            self.bm25_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        if self.doc_hashes_path.exists():
            self.doc_hashes_path.unlink()
        self.index = None
        self.bm25 = None
        self.tokenized_corpus = []
        self.meta = []
        self.doc_hashes = {}

    def load(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Index not found. Run ingest first.")
        self.index = faiss.read_index(str(self.index_path))
        self.meta = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))
        
        # Load BM25 index
        if self.bm25_path.exists():
            with self.bm25_path.open("rb") as f:
                bm25_data = pickle.load(f)
                self.tokenized_corpus = bm25_data["tokenized_corpus"]
                self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Load document hashes
        if self.doc_hashes_path.exists():
            with self.doc_hashes_path.open("r", encoding="utf-8") as f:
                self.doc_hashes = json.load(f)
        else:
            self.doc_hashes = {}

    def save(self):
        assert self.index is not None
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        
        # Save BM25 index
        if self.bm25 is not None and self.tokenized_corpus:
            with self.bm25_path.open("wb") as f:
                pickle.dump({"tokenized_corpus": self.tokenized_corpus}, f)
        
        # Save document hashes
        with self.doc_hashes_path.open("w", encoding="utf-8") as f:
            json.dump(self.doc_hashes, f, indent=2)
    
    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple tokenization for BM25"""
        return text.lower().split()

    def build(self, vectors: np.ndarray, chunks: list[Chunk]):
        if len(chunks) == 0:
            raise ValueError("No chunks to index.")

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity because embeddings normalized
        index.add(vectors)

        self.index = index
        self.meta = []
        self.tokenized_corpus = []
        
        for i, ch in enumerate(chunks):
            self.meta.append({
                "id": i,
                "source": ch.source,
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "timestamp": ch.timestamp,
                "images": ch.images,
            })
            self.tokenized_corpus.append(self.tokenize(ch.text))
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def add_documents(self, vectors: np.ndarray, chunks: list[Chunk], file_hash: str):
        """Add new documents incrementally"""
        if len(chunks) == 0:
            return
        
        if self.index is None:
            # First time - build from scratch
            self.build(vectors, chunks)
            return
        
        # Get the current max id
        current_max_id = max([m["id"] for m in self.meta]) if self.meta else -1
        
        # Add vectors to index
        self.index.add(vectors)
        
        # Add metadata and tokenized corpus
        for i, ch in enumerate(chunks):
            self.meta.append({
                "id": current_max_id + 1 + i,
                "source": ch.source,
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "timestamp": ch.timestamp,
                "images": ch.images,
            })
            self.tokenized_corpus.append(self.tokenize(ch.text))
        
        # Rebuild BM25 index (fast operation)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Update document hash
        if chunks:
            source = chunks[0].source
            self.doc_hashes[source] = file_hash

    def remove_document(self, source: str) -> bool:
        """Remove all chunks from a specific document and rebuild index"""
        if self.index is None or not self.meta:
            return False
        
        # Find indices to keep
        indices_to_keep = [i for i, m in enumerate(self.meta) if m["source"] != source]
        
        if len(indices_to_keep) == len(self.meta):
            # Document not found
            return False
        
        logger.info(f"🗑️  Removing {len(self.meta) - len(indices_to_keep)} chunks from '{source}'")
        
        # Rebuild index without removed documents
        if len(indices_to_keep) == 0:
            # All documents removed
            self.reset()
            self.save()
            return True
        
        # FAISS doesn't support removing vectors, so we rebuild the index
        # Extract vectors from the old index
        dim = self.index.d
        old_vectors = np.zeros((self.index.ntotal, dim), dtype=np.float32)
        for i in range(self.index.ntotal):
            old_vectors[i] = self.index.reconstruct(int(i))
        
        # Keep only the vectors we want
        vectors_to_keep = old_vectors[indices_to_keep]
        
        # Rebuild FAISS index
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(vectors_to_keep)
        self.index = new_index
        
        # Update metadata
        self.meta = [self.meta[i] for i in indices_to_keep]
        
        # Update IDs
        for new_id, m in enumerate(self.meta):
            m["id"] = new_id
        
        # Rebuild BM25 index
        self.tokenized_corpus = [self.tokenized_corpus[i] for i in indices_to_keep]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Remove from hashes
        if source in self.doc_hashes:
            del self.doc_hashes[source]
        
        return True

    def needs_update(self, source: str, file_hash: str) -> bool:
        """Check if document needs to be updated"""
        return self.doc_hashes.get(source) != file_hash
    
    def get_document_info(self, source: str) -> dict:
        """Get information about a document including latest timestamp"""
        chunks = [m for m in self.meta if m["source"] == source]
        if not chunks:
            return None
        
        latest_timestamp = max(c.get("timestamp", 0) for c in chunks)
        import datetime
        latest_dt = datetime.datetime.fromtimestamp(latest_timestamp)
        
        return {
            "source": source,
            "num_chunks": len(chunks),
            "latest_timestamp": latest_timestamp,
            "latest_datetime": latest_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "file_hash": self.doc_hashes.get(source, "unknown")
        }

    def search(self, query_vec: np.ndarray, top_k: int = 5, query_text: str = None, hybrid: bool = True, bm25_weight: float = 0.3) -> list[dict]:
        """
        Search with optional hybrid mode combining BM25 and vector search.
        
        Args:
            query_vec: Query embedding vector
            top_k: Number of results to return
            query_text: Original query text (required for hybrid search)
            hybrid: If True and query_text provided, use hybrid search
            bm25_weight: Weight for BM25 scores (0-1), vector gets (1-weight)
        """
        assert self.index is not None
        
        if hybrid and query_text and self.bm25 is not None:
            return self._hybrid_search(query_vec, query_text, top_k, bm25_weight)
        else:
            return self._vector_search(query_vec, top_k)
    
    def _vector_search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """Pure vector similarity search"""
        # Search with higher k to account for duplicates
        scores, ids = self.index.search(query_vec, min(top_k * 3, self.index.ntotal))
        
        # Build results with deduplication by source (keep latest version only)
        seen_sources = {}  # source -> (score, metadata)
        
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx == -1 or idx >= len(self.meta):
                continue
            
            m = dict(self.meta[idx])
            m["score"] = float(score)
            source = m["source"]
            timestamp = m.get("timestamp", 0)
            
            # Keep only the latest version of each document chunk
            if source not in seen_sources:
                seen_sources[source] = []
            
            seen_sources[source].append((timestamp, m))
        
        # Filter to latest versions and sort by score
        out = []
        for source, versions in seen_sources.items():
            # Get the latest version (highest timestamp)
            latest = max(versions, key=lambda x: x[0])[1]
            out.append(latest)
        
        # Sort by score descending and limit to top_k
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]
    
    def _hybrid_search(self, query_vec: np.ndarray, query_text: str, top_k: int, bm25_weight: float) -> list[dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF) to combine BM25 and vector search.
        RRF is more robust than score normalization and handles different score scales well.
        """
        # Get vector search results (more results for better fusion)
        k_retrieval = min(top_k * 5, self.index.ntotal)
        vector_scores, vector_ids = self.index.search(query_vec, k_retrieval)
        
        # Get BM25 scores
        tokenized_query = self.tokenize(query_text)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Build ranked lists
        vector_ranks = {}  # doc_id -> rank (0-indexed)
        for rank, (score, idx) in enumerate(zip(vector_scores[0], vector_ids[0])):
            if idx >= 0 and idx < len(self.meta):
                vector_ranks[int(idx)] = rank
        
        bm25_ranks = {}  # doc_id -> rank (0-indexed)
        bm25_sorted = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        for rank, (idx, score) in enumerate(bm25_sorted[:k_retrieval]):
            if idx < len(self.meta):
                bm25_ranks[idx] = rank
        
        # Reciprocal Rank Fusion: score = sum(1 / (k + rank))
        # k=60 is a common default that works well
        rrf_k = 60
        rrf_scores = {}
        
        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        for doc_id in all_doc_ids:
            vector_score = 1 / (rrf_k + vector_ranks.get(doc_id, k_retrieval)) if doc_id in vector_ranks else 0
            bm25_score = 1 / (rrf_k + bm25_ranks.get(doc_id, k_retrieval)) if doc_id in bm25_ranks else 0
            
            # Weighted combination
            rrf_scores[doc_id] = (1 - bm25_weight) * vector_score + bm25_weight * bm25_score
        
        # Sort by RRF score
        ranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build results with deduplication
        seen_sources = {}
        
        for doc_id, rrf_score in ranked_docs[:top_k * 3]:
            if doc_id >= len(self.meta):
                continue
            
            m = dict(self.meta[doc_id])
            m["score"] = float(rrf_score)
            m["vector_rank"] = vector_ranks.get(doc_id, -1)
            m["bm25_rank"] = bm25_ranks.get(doc_id, -1)
            
            source = m["source"]
            timestamp = m.get("timestamp", 0)
            
            if source not in seen_sources:
                seen_sources[source] = []
            
            seen_sources[source].append((timestamp, m))
        
        # Filter to latest versions
        out = []
        for source, versions in seen_sources.items():
            latest = max(versions, key=lambda x: x[0])[1]
            out.append(latest)
        
        # Already sorted by RRF score
        return out[:top_k]
