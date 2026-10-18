
def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_top_k = retrieved_ids[:k]
    correct = len(set(retrieved_top_k) & set(relevant_ids))
    return correct / k if k > 0 else 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_top_k = retrieved_ids[:k]
    correct = len(set(retrieved_top_k) & set(relevant_ids))
    return correct / len(relevant_ids) if relevant_ids else 0.0


def evaluate_retrieval(test_set, retrieve_fn, k=5):
    """
    test_set    -> list of {"query": ..., "relevant_ids": [...]}
    retrieve_fn -> ek function jo query leta hai aur chunk_ids ki list deta hai
    """
    precisions = []
    recalls = []
    per_query_results = []

    for item in test_set:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        retrieved_ids = retrieve_fn(query, k)

        p = precision_at_k(retrieved_ids, relevant_ids, k)
        r = recall_at_k(retrieved_ids, relevant_ids, k)

        precisions.append(p)
        recalls.append(r)
        per_query_results.append({
            "query": query, "precision": round(p, 2), "recall": round(r, 2),
            "retrieved": retrieved_ids, "relevant": relevant_ids,
        })

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    return avg_precision, avg_recall, per_query_results