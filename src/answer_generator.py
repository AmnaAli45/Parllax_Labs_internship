import time
from sentence_transformers import SentenceTransformer
from vector_store import get_collection, semantic_search
from prompt import build_prompt
from llm_clent import call_llm
from hallucination_guard import is_in_domain, is_refusal, OUT_OF_DOMAIN_MESSAGE


def answer_question(query, model, collection, top_k=3):
    total_start = time.time()
    # ------------------------------------------------- Retrieval timing --------------------------------------------------------
    retrieval_start = time.time()
    query_vector = model.encode(query).tolist()
    results = semantic_search(collection, query_vector, top_k=top_k)
    retrieval_ms = (time.time() - retrieval_start) * 1000
    chunks = results["documents"][0]
    distances = results["distances"][0]

    if not is_in_domain(distances):
        total_ms = (time.time() - total_start) * 1000
        return {
            "answer": OUT_OF_DOMAIN_MESSAGE,
            "status": "out_of_domain",
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": 0.0,         
            "total_ms": round(total_ms, 2),
        }

    # ------------------------------------------------- Generation timing --------------------------------------------------------
    generation_start = time.time()
    messages = build_prompt(query, chunks)
    answer, error = call_llm(messages)
    generation_ms = (time.time() - generation_start) * 1000

    total_ms = (time.time() - total_start) * 1000

    if error:
        return {
            "answer": f"Something went wrong: {error}",
            "status": "api_error",
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
        }

    status = "refused_in_domain" if is_refusal(answer) else "answered"

    return {
        "answer": answer,
        "status": status,
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "total_ms": round(total_ms, 2),
    }


def log_result(query, result):
    print(f"Query: {query}")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}")
    print(
        f"Timing -> retrieval: {result['retrieval_ms']}ms | "
        f"generation: {result['generation_ms']}ms | "
        f"total: {result['total_ms']}ms"
    )
    print()


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = get_collection()

    test_queries = [
        "What is deep learning?",
        "What's the best pizza topping?",
    ]

    for q in test_queries:
        result = answer_question(q, model, collection)
        log_result(q, result)