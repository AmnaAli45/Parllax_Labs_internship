"""
Poori RAG pipeline (retrieve + generate) ko automated test queries par chalata hai,
aur generation quality (keyword match) + latency record karta hai.
"""
import statistics


def check_quality(answer_text, expected_keywords):
    """
    Dekhta hai answer mein expected keywords mein se kitne mojood hain.
    Returns: (score 0 to 1, list of missing keywords)
    """
    if not expected_keywords:
        return None, []

    answer_lower = answer_text.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

    score = len(found) / len(expected_keywords)
    return score, missing


def run_e2e_evaluation(test_queries, answer_fn):
    
    results = []

    for item in test_queries:
        query = item["query"]
        expected_keywords = item.get("expected_keywords", [])

        result = answer_fn(query)

        quality_score, missing_keywords = check_quality(result["answer"], expected_keywords)

        results.append({
            "query": query,
            "status": result["status"],
            "answer": result["answer"],
            "quality_score": quality_score,
            "missing_keywords": missing_keywords,
            "retrieval_ms": result["retrieval_ms"],
            "generation_ms": result["generation_ms"],
            "total_ms": result["total_ms"],
        })

    return results


def summarize_results(results):
    """Latency aur quality ka overall summary nikalta hai."""
    status_counts = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    quality_scores = [r["quality_score"] for r in results if r["quality_score"] is not None]
    total_latencies = [r["total_ms"] for r in results]
    retrieval_latencies = [r["retrieval_ms"] for r in results]
    generation_latencies = [r["generation_ms"] for r in results]

    summary = {
        "total_queries": len(results),
        "status_breakdown": status_counts,
        "avg_quality_score": round(statistics.mean(quality_scores), 2) if quality_scores else None,
        "avg_total_latency_ms": round(statistics.mean(total_latencies), 2),
        "avg_retrieval_latency_ms": round(statistics.mean(retrieval_latencies), 2),
        "avg_generation_latency_ms": round(statistics.mean(generation_latencies), 2),
        "max_total_latency_ms": round(max(total_latencies), 2),
    }
    return summary