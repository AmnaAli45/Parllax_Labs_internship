from sentence_transformers import SentenceTransformer
from vector_store import get_collection, semantic_search
from retrieval_eval import evaluate_retrieval

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = get_collection()


def retrieve_fn(query, k):
    query_vector = model.encode(query).tolist()
    results = semantic_search(collection, query_vector, top_k=k)
    return results["ids"][0]



test_set = [
    {"query": "What is deep learning?", "relevant_ids": ["12_chunk0", "45_chunk2"]},
    {"query": "Who was Albert Einstein?", "relevant_ids": ["88_chunk1"]},
    
]

avg_p, avg_r, details = evaluate_retrieval(test_set, retrieve_fn, k=5)
print(f"Average Precision@5: {avg_p:.2f}")
print(f"Average Recall@5: {avg_r:.2f}")

for d in details:
    print(d)