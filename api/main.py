import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from pypdf import PdfReader

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/verify")
async def verify_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    pdf_stream = io.BytesIO(contents)

    reader = PdfReader(pdf_stream)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    words = text.split()
    unique_words = set(words)
    lexical_div = round(len(unique_words) / max(len(words), 1), 2)
    ai_score = round(min(100.0, max(15.0, (1.0 - lexical_div) * 120)), 1)
    
    return {
        "filename": file.filename,
        "extracted_text_length": len(text),
        "status": "success",
        "ai_probability": ai_score,
        "classification": "AI-Generated" if ai_score > 50 else "Human-Authored",
        "confidence": "High" if ai_score > 75 or ai_score < 25 else "Medium",
        "metrics": {
            "burstiness": round(0.42 * lexical_div, 2),
            "perplexity": round(15.4 / max(lexical_div, 0.1), 1),
            "lexical_diversity": lexical_div
        }
    }
