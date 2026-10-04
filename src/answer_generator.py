from sentence_transformers import SentenceTransformer
from src.vector_store import get_collection, semantic_search
from src.prompt import build_prompt

def retrieve_chunks(query, model, collection, top_k=3):
    query_vector = model.encode(query).tolist()
    results = semantic_search(collection, query_vector, top_k=top_k)
    return results["documents"][0]

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = get_collection()
    query = "What is deep learning?"
    chunks = retrieve_chunks(query, model, collection, top_k=3)
    print("Retrieved chunks:")
    for c in chunks:
        print(" -", c[:80])
    messages = build_prompt(query, chunks)
    print("\nPrompt ready. User message preview:")
    print(messages[1]["content"][:200])