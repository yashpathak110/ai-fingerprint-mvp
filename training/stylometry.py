import re
from typing import Dict

class StylometryExtractor:
    """Extracts explicit stylistic fingerprints from text."""
    
    FUNCTION_WORDS = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"}

    def extract_features(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
        total_words = max(len(tokens), 1)
        total_chars = max(len(text), 1)
        total_sentences = max(len(sentences), 1)

        avg_sentence_len = total_words / total_sentences
        avg_word_len = sum(len(w) for w in tokens) / total_words
        
        commas = text.count(",") / total_chars
        semicolons = text.count(";") / total_chars
        dashes = (text.count("-") + text.count("--")) / total_chars
        
        func_word_count = sum(1 for w in tokens if w in self.FUNCTION_WORDS)
        func_word_ratio = func_word_count / total_words

        return {
            "avg_sentence_len": round(avg_sentence_len, 4),
            "avg_word_len": round(avg_word_len, 4),
            "comma_density": round(commas, 4),
            "semicolon_density": round(semicolons, 4),
            "dash_density": round(dashes, 4),
            "func_word_ratio": round(func_word_ratio, 4)
        }