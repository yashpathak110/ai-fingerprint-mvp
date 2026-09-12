import requests

API_URL = "http://127.0.0.1:8001"

sample_docs = [
    {
        "id": "doc_agentic_001",
        "author": "Human_Architect",
        "text": "The platform implements a workflow-oriented agentic AI system where specialized functional stages handle resume parsing, job matching, interview generation, answer evaluation, and deterministic ranking."
    },
    {
        "id": "doc_llm_002",
        "author": "Claude-3.5-Sonnet",
        "text": "Large language models utilize transformer self-attention mechanisms to dynamically weight contextual token relationships across high-dimensional latent representations."
    }
]

def seed():
    for doc in sample_docs:
        res = requests.post(f"{API_URL}/ingest", json=doc)
        print(f"Ingested {doc['id']}: {res.status_code} | Chunks: {res.json().get('chunks_processed')}")

if __name__ == "__main__":
    seed()
