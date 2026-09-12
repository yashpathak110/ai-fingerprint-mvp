from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import os

class VectorDBManager:
    def __init__(self, collection_name="fingerprints", vector_size=384):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        qdrant_url = os.getenv("QDRANT_URL")
        
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
        else:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
        except Exception as e:
            print(f"Failed to initialize collection {self.collection_name}: {e}")

    def upsert_vectors(self, vectors, payloads):
        import uuid
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist() if hasattr(vec, 'tolist') else vec,
                payload=payload
            )
            for vec, payload in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_similar(self, query_vector, top_k=3):
        try:
            vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
            if isinstance(vector_list, list) and len(vector_list) > 0 and isinstance(vector_list[0], list):
                vector_list = vector_list[0]
                
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector_list,
                limit=top_k
            )
            
            formatted_results = []
            for hit in results:
                formatted_results.append({
                    "score": hit.score,
                    "author": hit.payload.get("author", "Unknown"),
                    "doc_id": hit.payload.get("doc_id", "Unknown"),
                    "text": hit.payload.get("text", "")
                })
            return formatted_results
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []