from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

class VectorDBManager:
    def __init__(self, collection_name="fingerprints", vector_size=384):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Connect to local/in-memory or remote Qdrant
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        
        # Fallback to local in-memory store if no remote URL provided
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
        from qdrant_client.http.models import PointStruct
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