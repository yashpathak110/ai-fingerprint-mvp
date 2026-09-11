import sys
import os
from typing import List

from data.loader import DataProcessor
from training.embedder import TextEmbedder
from vector_db.vector_manager import VectorDBManager

def run_ingestion(documents: List[dict]):
    """Pipeline to clean, chunk, embed, and store documents in Qdrant."""
    processor = DataProcessor(chunk_size=100, overlap=20)
    embedder = TextEmbedder()
    db = VectorDBManager()

    all_chunks = []
    all_payloads = []

    print("\n--- Processing Documents ---")
    for doc in documents:
        doc_id = doc["id"]
        author = doc.get("author", "unknown")
        chunks = processor.chunk_text(doc["text"], source_id=doc_id)
        
        for c in chunks:
            all_chunks.append(c["text"])
            all_payloads.append({
                "doc_id": doc_id,
                "chunk_id": c["chunk_id"],
                "author": author,
                "text": c["text"]
            })

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    vectors = embedder.embed_texts(all_chunks)

    # Assign sequential IDs for vector storage
    point_ids = list(range(1, len(vectors) + 1))
    
    print("Upserting vectors into Qdrant...")
    db.upsert_fingerprints(ids=point_ids, vectors=vectors, payloads=all_payloads)
    db.close()
    print("Ingestion pipeline completed successfully!")

if __name__ == "__main__":
    sample_corpus = [
        {
            "id": "doc_ai_01",
            "author": "GPT-4",
            "text": "Artificial intelligence models generate text by sampling from high-dimensional token probability distributions. These outputs retain subtle statistical signatures."
        },
        {
            "id": "doc_human_01",
            "author": "Human_Writer",
            "text": "I sat down at my desk early this morning, drinking hot coffee while reviewing hand-written notes from yesterday's workshop. The atmosphere was completely quiet."
        }
    ]
    run_ingestion(sample_corpus)