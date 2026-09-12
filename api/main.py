from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vector_db.vector_manager import VectorDBManager
from graph.nodes.app import FingerprintPipeline

app = FastAPI(title="AI-Fingerprint-MVP")

db = VectorDBManager()
pipeline = FingerprintPipeline(db_manager=db)

class IngestRequest(BaseModel):
    id: str
    author: str
    text: str

class SearchRequest(BaseModel):
    text: str
    top_k: int = 3

@app.get("/")
def health_check():
    return {"status": "online", "engine": "AI-Fingerprint-MVP"}

@app.post("/ingest")
def ingest_text(req: IngestRequest):
    try:
        from training.embedder import TextEmbedder
        embedder = TextEmbedder()
        vectors = embedder.embed_texts([req.text])
        payloads = [{"doc_id": req.id, "author": req.author, "text": req.text}]
        db.upsert_vectors(vectors, payloads)
        return {"status": "success", "chunks_processed": 1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_fingerprint(req: SearchRequest):
    try:
        verdict = pipeline.process(query_text=req.text, top_k=req.top_k)
        return {"verdict": {"final_verdict": verdict}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))