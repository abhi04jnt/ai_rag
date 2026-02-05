from __future__ import annotations
import re
import logging
from .config import settings
from .embeddings import Embedder
from .vectorstore import FaissStore
from .prompts import build_user_prompt
from .llm import get_llm

logger = logging.getLogger(__name__)

class ContentSafetyFilter:
    """Filter to detect PII, PHI, and inappropriate content"""
    
    # Patterns for common PII
    PII_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', 'Social Security Number'),  # SSN
        (r'\b\d{3}-\d{3}-\d{4}\b', 'Phone Number'),  # Phone
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email Address'),  # Email
        (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 'Credit Card Number'),  # Credit card
        (r'\b\d{5}(?:-\d{4})?\b', 'ZIP Code (if combined with other PII)'),  # ZIP
        (r'\b[A-Z]\d{2}[- ]?\d{6}\b', 'Passport Number'),  # Passport
    ]
    
    # Keywords indicating sensitive health/medical content
    PHI_KEYWORDS = [
        'patient', 'diagnosis', 'prescription', 'medical record', 'health insurance',
        'treatment plan', 'laboratory results', 'blood test', 'x-ray', 'mri scan',
        'hospital admission', 'doctor visit', 'medication', 'symptom', 'disease'
    ]
    
    # Keywords for adult/inappropriate content
    INAPPROPRIATE_KEYWORDS = [
        'explicit sexual', 'nsfw', 'adult content', 'pornography', 'pornographic'
    ]
    
    @staticmethod
    def check_content_safety(text: str) -> tuple[bool, str]:
        """
        Check if content is safe to process.
        Returns: (is_safe, reason)
        """
        text_lower = text.lower()
        
        # Check for PII patterns
        for pattern, pii_type in ContentSafetyFilter.PII_PATTERNS:
            if re.search(pattern, text):
                return False, f"Detected {pii_type}. This system cannot process personally identifiable information (PII) to comply with GDPR and privacy regulations."
        
        # Check for PHI keywords (multiple matches increase likelihood)
        phi_matches = sum(1 for keyword in ContentSafetyFilter.PHI_KEYWORDS if keyword in text_lower)
        if phi_matches >= 2:
            return False, "Detected protected health information (PHI). This system cannot process medical or health data to comply with HIPAA regulations."
        
        # Check for inappropriate content
        for keyword in ContentSafetyFilter.INAPPROPRIATE_KEYWORDS:
            if keyword in text_lower:
                return False, "Detected inappropriate content. This system maintains ethical AI guidelines and cannot process such material."
        
        return True, ""

class ChatService:
    def __init__(self):
        self.store = FaissStore(settings.index_dir)
        self.store.load()
        self.embedder = Embedder(settings.embedding_model)
        self.llm = get_llm()
        self.safety_filter = ContentSafetyFilter()
    
    async def _reformulate_with_context(self, question: str, history: list[dict]) -> str:
        """
        Use LLM to reformulate follow-up questions by adding context from conversation history.
        This is more robust than pattern matching.
        """
        if not history or len(history) < 2:
            # No context to add
            return question
        
        # Get recent conversation context
        recent_messages = history[-4:]  # Last 4 messages
        context_parts = []
        for msg in recent_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]  # Limit length
            if role == "user":
                context_parts.append(f"User asked: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant answered about: {content[:100]}")
        
        conversation_context = "\n".join(context_parts)
        
        # Check if this looks like a follow-up question (contains pronouns or vague references)
        vague_indicators = ['it', 'this', 'that', 'more', 'further', 'additional', 'the', 'them', 'they']
        first_words = question.lower().split()[:5]
        has_vague_reference = any(word in first_words for word in vague_indicators)
        
        # Also check for visual content requests
        visual_keywords = ['diagram', 'chart', 'graph', 'figure', 'image', 'picture', 'illustration', 'visualization', 'architecture']
        is_visual_request = any(keyword in question.lower() for keyword in visual_keywords)
        
        if not has_vague_reference and not is_visual_request:
            # Question is clear and standalone
            return question
        
        # Use LLM to reformulate the question with context
        reformulation_prompt = f"""Given this conversation history:

{conversation_context}

The user now asks: "{question}"

This question seems to be a follow-up that references the previous conversation. 
Reformulate this question to be standalone and clear by adding necessary context from the conversation history.

Rules:
- Make the question self-contained and specific
- Include the main topic being discussed
- Keep it concise (one sentence)
- If asking for visual content, specify what kind (e.g., "RAG architecture diagram" instead of just "the diagram")
- Only output the reformulated question, nothing else

Reformulated question:"""
        
        try:
            reformulated = await self.llm.complete(reformulation_prompt)
            reformulated = reformulated.strip().strip('"').strip("'")
            
            if reformulated and len(reformulated) > 10 and len(reformulated) < 200:
                logger.info(f"LLM reformulated query: '{question}' → '{reformulated}'")
                return reformulated
        except Exception as e:
            logger.warning(f"LLM reformulation failed: {e}, using original question")
        
        return question

    async def answer(self, question: str, history: list[dict]) -> dict:
        try:
            logger.info(f"Processing question: '{question}'")
            
            # Pre-screen the question for sensitive content
            is_safe, reason = self.safety_filter.check_content_safety(question)
            if not is_safe:
                return {
                    "answer": f"🚫 **Content Safety Alert**\n\n{reason}\n\n**This system is designed to:**\n• Protect privacy and comply with HIPAA, GDPR, and AI Acts\n• Refuse processing of PII, PHI, and sensitive personal data\n• Maintain ethical AI standards\n\nPlease rephrase your question without sensitive information.",
                    "retrieved": []
                }
            
            # Reformulate question with context from history for better retrieval
            search_query = await self._reformulate_with_context(question, history)
            if search_query != question:
                logger.info(f"Reformulated query: '{question}' → '{search_query}'")
            
            # Check if user is asking for visual content (diagrams, charts, images)
            visual_keywords = ['diagram', 'chart', 'graph', 'figure', 'image', 'picture', 'illustration', 'visualization', 'architecture']
            is_visual_query = any(keyword in search_query.lower() for keyword in visual_keywords)
            
            qvec = self.embedder.embed_texts([search_query])
            
            # Use hybrid search combining BM25 (keyword) + vector (semantic)
            # BM25 helps with exact term matches, vector search handles semantic similarity
            # Lower BM25 weight (0.15) favors semantic understanding for related terms
            retrieved = self.store.search(
                qvec, 
                top_k=settings.top_k * 2 if is_visual_query else settings.top_k,
                query_text=search_query,
                hybrid=False,  # Temporarily disabled - using pure vector search
                bm25_weight=0.15  # 15% BM25, 85% vector - better for semantic queries
            )
            
            logger.info(f"Hybrid search retrieved {len(retrieved)} chunks")
            
            # Log top 3 retrieved chunks for debugging
            for idx, chunk in enumerate(retrieved[:3], 1):
                logger.info(f"  Chunk {idx}: {chunk['source']} (score: {chunk['score']:.3f}) - {chunk['text'][:100]}...")
            
            # If user is asking for visual content, prioritize chunks with images
            if is_visual_query:
                logger.info(f"Visual query - retrieved {len(retrieved)} total chunks")
                # Separate chunks with and without images
                chunks_with_images = [r for r in retrieved if r.get('images') and len(r.get('images', [])) > 0]
                chunks_without_images = [r for r in retrieved if not r.get('images') or len(r.get('images', [])) == 0]
                
                logger.info(f"  - {len(chunks_with_images)} chunks WITH images")
                logger.info(f"  - {len(chunks_without_images)} chunks WITHOUT images")
                
                # If no images found but user wants diagrams, try getting chunks from image-rich documents
                if len(chunks_with_images) == 0 and 'architecture' in search_query.lower():
                    logger.info("No images found - searching all chunks from RAG ebook for images")
                    # Get all chunks from the RAG ebook that have images
                    from .vectorstore import FaissStore
                    all_chunks_with_images = []
                    for i, meta in enumerate(self.store.meta):
                        if 'building-blocks-of-rag' in meta.get('source', '').lower() and meta.get('images') and len(meta.get('images', [])) > 0:
                            all_chunks_with_images.append({
                                'id': i,
                                'source': meta['source'],
                                'chunk_id': meta['chunk_id'],
                                'text': meta['text'],
                                'score': 0.5,  # Moderate score
                                'images': meta['images']
                            })
                    
                    if all_chunks_with_images:
                        logger.info(f"  - Found {len(all_chunks_with_images)} image chunks from RAG ebook")
                        chunks_with_images = all_chunks_with_images[:settings.top_k]
                
                if chunks_with_images:
                    for idx, chunk in enumerate(chunks_with_images[:3]):
                        logger.info(f"  - Image chunk {idx+1}: {chunk['source']} chunk#{chunk['chunk_id']} ({len(chunk.get('images', []))} images)")
                
                # Prioritize chunks with images, then add text-only chunks
                retrieved = chunks_with_images[:settings.top_k] + chunks_without_images[:max(1, settings.top_k - len(chunks_with_images))]
                retrieved = retrieved[:settings.top_k]
                
                logger.info(f"Final: prioritized {len([r for r in retrieved if r.get('images')])} chunks with images")

            prompt = build_user_prompt(question=question, retrieved=retrieved, history=history)
            text = await self.llm.complete(prompt)
            
            # Check if we got a valid response
            if not text or len(text.strip()) < 10:
                logger.warning(f"Empty or very short LLM response: '{text}'")
                text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

            return {
                "answer": text,
                "retrieved": [
                    {
                        "source": r["source"],
                        "chunk_id": r["chunk_id"],
                        "score": r["score"],
                        "text": r["text"][:400] + ("..." if len(r["text"]) > 400 else ""),
                        "images": r.get("images", [])
                    }
                    for r in retrieved
                ],
            }
        except RuntimeError as e:
            # Re-raise with the friendly error message
            logger.error(f"RuntimeError in answer(): {e}")
            raise
        except Exception as e:
            logger.error(f"Exception in answer(): {e}", exc_info=True)
            raise RuntimeError(f"Error processing question: {str(e)}") from e
