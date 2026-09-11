from sentence_transformers import SentenceTransformer
from typing import List

class TextEmbedder:
    """Handles vector embedding generation for AI text fingerprinting."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the Sentence Transformer model."""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of text strings."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

if __name__ == "__main__":
    # Quick sanity check
    embedder = TextEmbedder()
    sample_texts = [
        "Artificial intelligence text fingerprinting MVP.",
        "Testing vector embeddings generation pipeline."
    ]
    vectors = embedder.embed_texts(sample_texts)
    print(f"Generated {len(vectors)} vectors.")
    print(f"Vector dimension size: {len(vectors[0])}")