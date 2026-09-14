import os
import io
import uvicorn
import joblib
import numpy as np
import spacy
import textstat
import pypdf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Fingerprint Engine API")

# Explicit CORS settings to allow GitHub Pages access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NLP & Machine Learning Models
nlp = spacy.load("en_core_web_sm")
model = joblib.load("ai_fingerprint_model.pkl")

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB Limit

def extract_features_and_metrics(text: str):
    doc = nlp(text)
    
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    sent_lengths = [len(s.split()) for s in sentences]
    
    sent_len_std = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    avg_sent_len = float(np.mean(sent_lengths)) if len(sent_lengths) > 0 else float(len(text.split()))

    tokens = [token.text.lower() for token in doc if token.is_alpha]
    ttr = float(len(set(tokens)) / len(tokens)) if len(tokens) > 0 else 0.0

    flesch_score = float(textstat.flesch_reading_ease(text)) if text.strip() else 0.0
    punct_count = sum(1 for char in text if char in "!?,.;:")
    punct_density = float(punct_count / len(text)) if len(text) > 0 else 0.0
    
    stopword_count = sum(1 for token in doc if token.is_stop)
    stopword_ratio = float(stopword_count / len(tokens)) if len(tokens) > 0 else 0.0

    features = [avg_sent_len, sent_len_std, ttr, flesch_score, punct_density, stopword_ratio]
    
    short_sents = sum(1 for l in sent_lengths if l <= 10)
    med_sents = sum(1 for l in sent_lengths if 11 <= l <= 25)
    long_sents = sum(1 for l in sent_lengths if l > 25)

    return features, {
        "sentence_distribution": [short_sents, med_sents, long_sents],
        "sentence_lengths_sample": sent_lengths[:20]
    }

@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Fingerprint Engine API active."}

@app.post("/analyze")
async def analyze_text(
    text: str = Form(default=""),
    file: UploadFile = File(default=None)
):
    extracted_text = ""

    if file and file.filename:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            return {"error": "File size exceeds 200 MB limit."}
        
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            max_pages = min(len(pdf_reader.pages), 100)
            for i in range(max_pages):
                t = pdf_reader.pages[i].extract_text()
                if t:
                    extracted_text += t + "\n"
        except Exception as e:
            return {"error": f"Failed to parse PDF file: {str(e)}"}

    if not extracted_text.strip() and text.strip():
        extracted_text = text

    if not extracted_text.strip():
        return {"error": "Could not extract readable text from input or file."}

    words = extracted_text.split()
    if len(words) > 100000:
        extracted_text = " ".join(words[:100000])

    features, extra_viz = extract_features_and_metrics(extracted_text)
    prediction = model.predict([features])[0]
    ai_prob = model.predict_proba([features])[0][1] * 100

    return {
        "text_preview": extracted_text[:300] + "...",
        "word_count": len(words),
        "is_ai": bool(prediction == 1),
        "ai_probability": round(float(ai_prob), 2),
        "human_probability": round(float(100 - ai_prob), 2),
        "metrics": {
            "burstiness": round(features[1], 2),
            "vocab_diversity": round(features[2] * 100, 2),
            "readability": round(features[3], 2),
            "punct_density": round(features[4] * 100, 2),
            "stopword_ratio": round(features[5] * 100, 2),
            "avg_sent_length": round(features[0], 2)
        },
        "viz": extra_viz
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)