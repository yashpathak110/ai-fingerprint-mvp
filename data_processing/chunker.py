import re
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return []
        
        words = cleaned.split(" ")
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += (self.chunk_size - self.overlap)
            
        return chunks
