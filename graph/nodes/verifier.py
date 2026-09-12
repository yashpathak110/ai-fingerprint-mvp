import os
import numpy as np
import xgboost as xgb
import spacy

nlp = spacy.load("en_core_web_sm")

class VerifierNode:
    def __init__(self, threshold: float = 0.70, model_path: str = "models/stylometry_xgb.json", **kwargs):
        self.confidence_threshold = threshold
        self.model_path = model_path
        self.xgb_model = None
        
        if os.path.exists(self.model_path):
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(self.model_path)

    def extract_stylometry(self, text: str) -> np.ndarray:
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        sentences = list(doc.sents)
        
        if not tokens or not sentences:
            return np.zeros((1, 4))

        # 1. Type-Token Ratio (TTR)
        ttr = len(set(tokens)) / len(tokens)
        
        # 2. Average Sentence Length
        avg_sent_len = len(tokens) / len(sentences)
        
        # 3. Burstiness (Sentence Length Variance)
        sent_lengths = [len([t for t in sent if t.is_alpha]) for sent in sentences]
        burstiness = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
        
        # 4. Punctuation Density
        punct_count = sum(1 for token in doc if token.is_punct)
        punct_density = punct_count / len(doc) if len(doc) > 0 else 0.0
        
        return np.array([[ttr, avg_sent_len, burstiness, punct_density]])

    def run(self, state: dict) -> dict:
        classification = state.get("classification", "Unknown")
        confidence = state.get("confidence_score", 0.0)
        query_text = state.get("query_text", "")

        ai_probability = 0.5
        if self.xgb_model is not None and query_text:
            features = self.extract_stylometry(query_text)
            dmatrix = xgb.DMatrix(features)
            ai_probability = float(self.xgb_model.predict(dmatrix)[0])

        # Hybrid Decision: Weighted blend of vector similarity and stylometric confidence
        hybrid_score = (0.6 * confidence) + (0.4 * (1.0 - ai_probability if classification == "Human_Architect" else ai_probability))

        if hybrid_score < self.confidence_threshold:
            verified_author = "Unknown"
            status = "Low Hybrid Confidence"
        else:
            verified_author = classification
            status = "Verified"

        state["verified_result"] = {
            "author": verified_author,
            "score": round(hybrid_score, 4),
            "vector_score": round(confidence, 4),
            "ai_probability": round(ai_probability, 4),
            "status": status
        }
        return state
