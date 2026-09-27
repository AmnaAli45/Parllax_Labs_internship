import time
from sentence_transformers import SentenceTransformer


def load_model(model_name="all-MiniLM-L6-v2"):
    start = time.time()
    model = SentenceTransformer(model_name)
    print(f"Model loaded in {time.time() - start:.2f} seconds")
    return model

def embed_texts(model, texts, batch_size=32):
    start = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    total_time = time.time() - start
    avg_time = total_time / len(texts) if texts else 0
    print(f"Done. Total time: {total_time:.2f}s | Avg per text: {avg_time*1000:.2f}ms "
          f"| Speed: {len(texts)/total_time:.1f} texts/sec")
    return embeddings


