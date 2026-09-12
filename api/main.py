from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vector_db.vector_manager import VectorDBManager
from graph.nodes.app import FingerprintPipeline
import spacy
import numpy as np

app = FastAPI(title="AI-Fingerprint-MVP")

db = VectorDBManager()
pipeline = FingerprintPipeline(db_manager=db)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import en_core_web_sm
    nlp = en_core_web_sm.load()

class IngestRequest(BaseModel):
    id: str
    author: str
    text: str

class SearchRequest(BaseModel):
    text: str
    top_k: int = 3

def calculate_ai_probability(text: str):
    """Fallback standalone stylometric statistical model for unseen AI text."""
    doc = nlp(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    sentences = list(doc.sents)
    
    if not tokens or not sentences:
        return 0.5, "Insufficient Text"

    ttr = len(set(tokens)) / len(tokens)
    avg_sentence_len = len(tokens) / len(sentences)
    lengths = [len([t for t in s if t.is_alpha]) for s in sentences]
    burstiness = float(np.std(lengths)) if len(lengths) > 1 else 0.0

    # LLMs tend to have lower burstiness (< 4.5) and predictable sentence length (~15-22 wps)
    ai_score = 0.0
    if burstiness < 4.5:
        ai_score += 0.40
    if 14 <= avg_sentence_len <= 24:
        ai_score += 0.35
    if ttr < 0.45:
        ai_score += 0.25

    return round(ai_score, 2)

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
        
        # Calculate statistical AI probability fallback
        ai_prob = calculate_ai_probability(req.text)
        
        author = verdict.get("author", "Unknown")
        score = verdict.get("score", 0.0)

        # Determine overall classification
        if score > 0.70:
            classification = f"Matched Database ({author})"
            is_ai = "AI Generated" if "ai" in author.lower() or "gpt" in author.lower() else "Human Written"
        else:
            is_ai = "AI Generated Signature" if ai_prob >= 0.60 else "Human Written Signature"
            classification = f"Statistical Signature ({is_ai})"

        return {
            "verdict": {
                "final_verdict": {
                    "author": author if score > 0.70 else is_ai,
                    "score": max(score, ai_prob),
                    "ai_probability": ai_prob,
                    "vector_score": score,
                    "status": classification
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))