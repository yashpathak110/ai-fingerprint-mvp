from graph.nodes.retriever import RetrieverNode
from graph.nodes.classifier import ClassifierNode
from graph.nodes.verifier import VerifierNode
from graph.nodes.aggregator import AggregatorNode
from vector_db.vector_manager import VectorDBManager

class FingerprintPipeline:
    def __init__(self, threshold: float = 0.30, db_manager: VectorDBManager = None):
        self.db = db_manager or VectorDBManager()
        self.retriever = RetrieverNode(db_manager=self.db)
        self.classifier = ClassifierNode()
        self.verifier = VerifierNode(threshold=threshold)
        self.aggregator = AggregatorNode()

    def process(self, query_text: str, top_k: int = 3) -> dict:
        state = {
            "query_text": query_text,
            "top_k": top_k,
            "matches": [],
            "classification": None,
            "confidence_score": 0.0,
            "is_verified": False,
            "final_verdict": {}
        }

        state = self.retriever.run(state)
        state = self.classifier.run(state)
        state = self.verifier.run(state)
        state = self.aggregator.run(state)

        return state["final_verdict"]
