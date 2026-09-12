import re
import numpy as np
from typing import Dict, Any

class StylometryExtractor:
    def extract_features(self, text: str) -> Dict[str, Any]:
        words = re.findall(r'\w+', text.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        total_words = len(words)
        total_sentences = len(sentences)
        
        if total_words == 0 or total_sentences == 0:
            return {
                "avg_sentence_length": 0.0,
                "sentence_length_variance": 0.0,
                "type_token_ratio": 0.0,
                "punctuation_density": 0.0
            }

        # 1. Burstiness (Sentence Length Distribution & Variance)
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = float(np.mean(sentence_lengths))
        sent_len_var = float(np.var(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0

        # 2. Lexical Richness (Type-Token Ratio)
        unique_words = set(words)
        ttr = float(len(unique_words) / total_words)

        # 3. Punctuation Density
        punctuation_count = len(re.findall(r'[,;:\-"\']', text))
        punct_density = float(punctuation_count / total_words)

        return {
            "avg_sentence_length": round(avg_sent_len, 4),
            "sentence_length_variance": round(sent_len_var, 4),
            "type_token_ratio": round(ttr, 4),
            "punctuation_density": round(punct_density, 4)
        }
