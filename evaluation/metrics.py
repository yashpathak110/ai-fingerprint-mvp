import requests

BASE_URL = "http://127.0.0.1:8001"

# Includes exact matches, paraphrased variations, and completely un-seeded text
test_cases = [
    {
        "name": "Exact Match (Human)",
        "query": "The platform implements a workflow-oriented agentic AI system where specialized functional stages handle resume parsing",
        "expected_author": "Human_Architect"
    },
    {
        "name": "Paraphrased / Humanized (Human)",
        "query": "Our system uses an agentic AI architecture designed to parse resumes and match job descriptions step by step.",
        "expected_author": "Human_Architect"
    },
    {
        "name": "Exact Match (AI)",
        "query": "Large language models utilize transformer self-attention mechanisms to dynamically weight contextual token relationships",
        "expected_author": "Claude-3.5-Sonnet"
    },
    {
        "name": "Paraphrased (AI)",
        "query": "Transformer models apply self-attention to calculate token weights dynamically across high-dimensional vectors.",
        "expected_author": "Claude-3.5-Sonnet"
    },
    {
        "name": "Unseen Out-of-Distribution Text",
        "query": "The quick brown fox jumps over the lazy dog in a remote rural field near the mountains.",
        "expected_author": "Unknown" # Should trigger low confidence / threshold fallback
    }
]

def run_evaluation():
    passed = 0
    total = len(test_cases)
    
    print("--- Running Adversarial Evaluation Suite ---\n")
    for case in test_cases:
        response = requests.post(
            f"{BASE_URL}/search",
            json={"text": case["query"], "top_k": 3}
        )
        data = response.json().get("verdict", {})
        final_verdict = data.get("final_verdict", {})
        predicted_author = final_verdict.get("author")
        score = final_verdict.get("score")
        
        is_correct = (predicted_author == case["expected_author"])
        if is_correct:
            passed += 1
            
        print(f"[{case['name']}]")
        print(f"  Expected: {case['expected_author']} | Predicted: {predicted_author} | Score: {score} | Pass: {is_correct}\n")

    accuracy = (passed / total) * 100
    print("-------------------------------------------")
    print(f"Final Real-World Accuracy: {accuracy:.2f}% ({passed}/{total} Passed)")

if __name__ == "__main__":
    run_evaluation()
