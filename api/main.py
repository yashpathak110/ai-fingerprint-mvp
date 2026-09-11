from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from data.loader import DataProcessor
from training.embedder import TextEmbedder
from vector_db.vector_manager import VectorDBManager

app = FastAPI(
    title="AI Fingerprint MVP API",
    description="Local-first API for text embedding, ingestion, and fingerprint similarity search.",
    version="1.0.0"
)

embedder = TextEmbedder()
processor = DataProcessor(chunk_size=100, overlap=20)

class IngestRequest(BaseModel):
    id: str
    author: Optional[str] = "unknown"
    text: str

class SearchRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3

@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Fingerprint API running local-first."}

@app.post("/ingest")
def ingest_document(payload: IngestRequest):
    try:
        db = VectorDBManager()
        chunks = processor.chunk_text(payload.text, source_id=payload.id)
        
        texts = [c["text"] for c in chunks]
        vectors = embedder.embed_texts(texts)
        
        ids = [abs(hash(c["chunk_id"])) % (10**8) for c in chunks]
        payloads = [
            {
                "doc_id": payload.id,
                "chunk_id": c["chunk_id"],
                "author": payload.author,
                "text": c["text"]
            }
            for c in chunks
        ]
        
        db.upsert_fingerprints(ids=ids, vectors=vectors, payloads=payloads)
        db.close()
        
        return {
            "status": "success",
            "doc_id": payload.id,
            "chunks_processed": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_fingerprint(payload: SearchRequest):
    try:
        db = VectorDBManager()
        query_vector = embedder.embed_texts([payload.text])[0]
        matches = db.search_similar(query_vector=query_vector, top_k=payload.top_k)
        db.close()
        
        return {
            "status": "success",
            "query": payload.text,
            "matches": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
