import torch

# Force single-threaded CPU execution and zero grad memory
torch.set_num_threads(1)
torch.set_grad_enabled(False)

class TextEmbedder:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            # Ultra-lightweight model to prevent Render 512MB RAM crash on inference
            cls._model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        return cls._model

    def embed_texts(self, texts):
        model = self.get_model()
        with torch.no_grad():
            return model.encode(texts, convert_to_numpy=True)