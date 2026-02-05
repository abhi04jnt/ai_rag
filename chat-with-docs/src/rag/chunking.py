from dataclasses import dataclass, field
import time

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    timestamp: float = None
    images: list[dict] = field(default_factory=list)  # List of image metadata
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

def chunk_text(text: str, source: str, chunk_size: int = 1200, overlap: int = 200, images: list[dict] = None) -> list[Chunk]:
    # Simple character-based chunking (fast + predictable)
    # Larger chunks with more overlap for better context and retrieval
    text = text.replace("\x00", " ").strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    cid = 0
    images = images or []
    
    # Sort images by text position
    sorted_images = sorted(images, key=lambda x: x.get('text_position', 0))
    
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text_part = text[start:end].strip()
        if chunk_text_part:
            # Assign images to chunks based on position
            # Images belong to the first chunk that covers their position
            chunk_images = []
            for img in sorted_images:
                img_pos = img.get('text_position', 0)
                # If image position falls within this chunk's range in the original text
                if start <= img_pos * (len(text) / max(1, len(sorted_images))) <= end:
                    chunk_images.append(img)
            
            chunks.append(Chunk(text=chunk_text_part, source=source, chunk_id=cid, images=chunk_images))
            cid += 1
        start = max(end - overlap, end)
    
    # If no chunks got images via position, attach to first chunk
    if images and not any(c.images for c in chunks):
        if chunks:
            chunks[0].images = images
    
    return chunks
