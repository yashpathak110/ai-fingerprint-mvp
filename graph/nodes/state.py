from typing import TypedDict, List, Dict, Any, Optional

class PipelineState(TypedDict):
    query_text: str
    top_k: int
    matches: List[Dict[str, Any]]
    classification: Optional[str]
    confidence_score: float
    is_verified: bool
    final_verdict: Dict[str, Any]
