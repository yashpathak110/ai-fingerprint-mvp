import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorDBManager:
    _instance = None

    def __new__(cls, storage_path: str = "./data/qdrant_storage", collection_name: str = "fingerprints"):
        if cls._instance is None:
            cls._instance = super(VectorDBManager, cls).__new__(cls)
            cls._instance.storage_path = storage_path
            cls._instance.collection_name = collection_name
            cls._instance._client = None
        return cls._instance

    @property
    def client(self):
        if self._client is None:
            os.makedirs(self.storage_path, exist_ok=True)
            self._client = QdrantClient(path=self.storage_path)
            self._init_collection()
        return self._client

    def _init_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def upsert_vectors(self, vectors, payloads):
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload=pay)
            for vec, pay in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_similar(self, query_vector, top_k: int = 3):
        if hasattr(self.client, "query_points"):
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            ).points
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
        return [{"score": float(res.score), "payload": res.payload} for res in results]
