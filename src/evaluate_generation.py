from sentence_transformers import SentenceTransformer
from src.vector_store import get_collection
from src.generate_answer import answer_question
from src.generation_eval import run_e2e_evaluation, summarize_results

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = get_collection()

def answer_fn(query):
    return answer_question(query, model, collection)

test_queries = [
    {"query": "What is deep learning?", "expected_keywords": ["neural network"]},
    {"query": "Who was Albert Einstein?", "expected_keywords": ["physicist"]},
    {"query": "What's the best pizza topping?", "expected_keywords": []},  
   
]

results = run_e2e_evaluation(test_queries, answer_fn)
summary = summarize_results(results)

print(summary)
for r in results:
    print(r)