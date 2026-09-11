import os
import re
from typing import List, Dict, Any

class DataProcessor:
    """Handles text cleaning, sliding-window chunking, and metadata extraction."""

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text: str) -> str:
        """Normalizes whitespace and strips non-printable characters."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, source_id: str = "doc_1") -> List[Dict[str, Any]]:
        """Splits text into overlapping word chunks with metadata."""
        cleaned = self.clean_text(text)
        words = cleaned.split(" ")
        
        if len(words) <= self.chunk_size:
            return [{
                "chunk_id": f"{source_id}_0",
                "text": cleaned,
                "word_count": len(words)
            }]

        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{source_id}_{len(chunks)}",
                "text": chunk_text,
                "word_count": len(chunk_words)
            })
            if i + self.chunk_size >= len(words):
                break

        return chunks

if __name__ == "__main__":
    processor = DataProcessor(chunk_size=15, overlap=5)
    sample_doc = (
        "Large language models exhibit distinct probabilistic patterns in token generation. "
        "By analyzing sliding windows of text embeddings, an AI fingerprint can be constructed "
        "to differentiate synthetic content from human author style with high fidelity."
    )
    chunks = processor.chunk_text(sample_doc, source_id="test_doc")
    print(f"Generated {len(chunks)} chunks:\n")
    for c in chunks:
        print(f"[{c['chunk_id']}] ({c['word_count']} words): {c['text']}")