class ClassifierNode:
    def run(self, state: dict) -> dict:
        matches = state.get("matches", [])
        if matches and len(matches) > 0:
            top_match = matches[0]
            payload = top_match.get("payload", {})
            state["classification"] = payload.get("author", "Unknown")
            state["confidence_score"] = float(top_match.get("score", 0.0))
        else:
            state["classification"] = "Unknown"
            state["confidence_score"] = 0.0
        return state
