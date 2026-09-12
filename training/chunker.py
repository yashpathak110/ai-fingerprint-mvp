import pymupdf
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        extracted_pages = []
        
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                extracted_pages.append(text.strip())
                
        doc.close()
        return "\n\n".join(extracted_pages)

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
            
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
