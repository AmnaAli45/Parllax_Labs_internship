import time
import statistics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentence_transformers import SentenceTransformer
from src.vector_store import get_collection, semantic_search


def test_queries(collection, model, queries, top_k=3):
    results_log = []
    for query in queries:
        start = time.time()
        query_vector = model.encode(query).tolist()         
        results = semantic_search(collection, query_vector, top_k=top_k)
        latency_ms = (time.time() - start) * 1000           
        results_log.append({
            "query": query,
            "latency_ms": latency_ms,
            "top_result": results["documents"][0][0] if results["documents"][0] else None,
            "distance": results["distances"][0][0] if results["distances"][0] else None,
        })
        print(f"Query:{query!r}")
        print(f" Latency:{latency_ms:.2f} ms")
        print(f"Top result:{results['documents'][0][0][:80] if results['documents'][0] else 'NONE'}")
        print(f"Distance: {results['distances'][0][0]:.4f}" if results["distances"][0] else "  Distance: N/A")
        print()

    return results_log


def print_summary(results_log):
    latencies = [r["latency_ms"] for r in results_log]
    print(f"Total queries : {len(latencies)}")
    print(f"Average       : {statistics.mean(latencies):.2f} ms")
    print(f"Min           : {min(latencies):.2f} ms")
    print(f"Max           : {max(latencies):.2f} ms")

if __name__ == "__main__":
    collection = get_collection()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    test_queries_list = [
        "What is machine learning?",
        "How do neural networks work?",
        "Tell me about healthy fruits.",
        "What is the capital of France?",
    ]
    log = test_queries(collection, model, test_queries_list, top_k=3)
    print_summary(log)