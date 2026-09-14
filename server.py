from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os # Added for Render Port handling
import uvicorn # Added to handle launching within the script

# ... (rest of your FastAPI setup above)

# The /analyze route setup...
@app.post("/analyze")
async def analyze_text(text: str = Form(default=""), file: UploadFile = File(default=None)):
    # ... (rest of analyze route logic)
    return {"status": "success"} # Placeholder

if __name__ == "__main__":
    # Get port assigned by Render, defaulting to 10000 if local
    port = int(os.environ.get("PORT", 10000))
    # Run Uvicorn listening on the correct PORT variable
    uvicorn.run(app, host="0.0.0.0", port=port)