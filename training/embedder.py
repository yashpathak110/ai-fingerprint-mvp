import torch

# Force single-threaded CPU execution to save RAM
torch.set_num_threads(1)

class TextEmbedder:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            # Load lightweight 384-dim model
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model

    def embed_texts(self, texts):
        model = self.get_model()
        return model.encode(texts, convert_to_numpy=True)