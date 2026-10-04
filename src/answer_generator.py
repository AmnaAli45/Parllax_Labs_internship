from sentence_transformers import SentenceTransformer
from vector_store import get_collection, semantic_search
from prompt import build_prompt
from llm_clent import call_llm
from hallucination_guard import is_in_domain, is_refusal, OUT_OF_DOMAIN_MESSAGE

def answer_question(query, model, collection, top_k=3):
    query_vector = model.encode(query).tolist()
    results = semantic_search(collection, query_vector, top_k=top_k)
    chunks = results["documents"][0]
    distances = results["distances"][0]

    if not is_in_domain(distances):
        return OUT_OF_DOMAIN_MESSAGE, "out_of_domain"

    messages = build_prompt(query, chunks)
    answer, error = call_llm(messages)

    if error:
        return f"Something went wrong: {error}", "api_error"

    if is_refusal(answer):
        return answer, "refused_in_domain"

    return answer, "answered"


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = get_collection()

    test_queries = [
        "What is deep learning?",                
        "What's the best pizza topping?",         
    ]

    for q in test_queries:
        answer, status = answer_question(q, model, collection)
        print(f"Query: {q}")
        print(f"Status: {status}")
        print(f"Answer: {answer}\n")