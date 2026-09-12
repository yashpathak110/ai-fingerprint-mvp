class AggregatorNode:
    def run(self, state: dict) -> dict:
        verified_result = state.get("verified_result", {})
        matches = state.get("matches", [])

        state["final_verdict"] = {
            "author": verified_result.get("author", "Unknown"),
            "score": verified_result.get("score", 0.0),
            "status": verified_result.get("status", "Unverified"),
            "top_matches_count": len(matches)
        }
        return state
