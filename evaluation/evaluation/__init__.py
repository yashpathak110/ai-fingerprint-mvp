import os
import sys
from typing import List, Dict, Any

from training.embedder import TextEmbedder
from vector_db.vector_manager import VectorDBManager

class FingerprintEvaluator:
    """Evaluates vector similarity thresholds and author attribution accuracy."""

    def __init__(self, similarity_threshold: float = 0.45):
        self.embedder = TextEmbedder()
        self.db = VectorDBManager()
        self.threshold = similarity_threshold

    def evaluate_queries(self, test_cases: List[Dict[str, Any]]):
        """Runs test queries against Qdrant storage and calculates attribution metrics."""
        correct_attributions = 0
        total_queries = len(test_cases)

        print("\n--- Running Fingerprint Attribution Benchmark ---")
        for idx, test in enumerate(test_cases, start=1):
            query_text = test["query"]
            expected_author = test["expected_author"]

            query_vector = self.embedder.embed_texts([query_text])[0]
            matches = self.db.search_similar(query_vector=query_vector, top_k=1)

            if not matches:
                predicted_author = "unknown"
                score = 0.0
            else:
                top_match = matches[0]
                score = top_match["score"]
                predicted_author = top_match["payload"].get("author", "unknown") if score >= self.threshold else "unknown"

            is_correct = predicted_author == expected_author
            if is_correct:
                correct_attributions += 1

            status_icon = "✓" if is_correct else "✗"
            print(f"[{status_icon}] Test {idx}: Score={score:.4f} | Pred={predicted_author} | Expected={expected_author}")

        accuracy = (correct_attributions / total_queries) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct_attributions}/{total_queries} correct)")
        self.db.close()

if __name__ == "__main__":
    benchmark_cases = [
        {
            "query": "How do transformer neural networks sample probabilities?",
            "expected_author": "Claude-3.5-Sonnet"
        },
        {
            "query": "I woke up early, made coffee, and worked on my notes.",
            "expected_author": "Human_Writer"
        }
    ]

    evaluator = FingerprintEvaluator(similarity_threshold=0.40)
    evaluator.evaluate_queries(benchmark_cases)