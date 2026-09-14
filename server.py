import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app FIRST before any route decorators
app = FastAPI(title="AI Fingerprint Engine API")

# Enable CORS so GitHub Pages frontend can access this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Fingerprint Engine is running"}

@app.post("/analyze")
async def analyze_text(text: str = Form(default=""), file: UploadFile = File(default=None)):
    try:
        # Dummy baseline response to verify endpoint connectivity
        word_count = len(text.split()) if text else 0
        return {
            "status": "success",
            "parsed_words": word_count,
            "classification": {
                "label": "HUMAN WRITTEN",
                "human_confidence": 60,
                "human_prob": 60,
                "ai_prob": 40,
                "burstiness": 11.64,
                "vocab_diversity": 17.17,
                "readability": 60.4,
                "avg_sentence_length": 18.86
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)