from training.embedder import TextEmbedder
from vector_db.vector_manager import VectorDBManager

class RetrieverNode:
    def __init__(self, db_manager: VectorDBManager = None):
        self.embedder = TextEmbedder()
        self.db = db_manager

    def run(self, state: dict) -> dict:
        if self.db is None:
            self.db = VectorDBManager()
            
        query_text = state.get("query_text", "")
        top_k = state.get("top_k", 3)
        
        query_vector = self.embedder.embed_texts([query_text])[0]
        matches = self.db.search_similar(query_vector=query_vector, top_k=top_k)
        
        state["matches"] = matches
        return state
