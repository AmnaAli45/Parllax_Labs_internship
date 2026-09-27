import time
from sentence_transformers import SentenceTransformer
import json


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
    print(f"Done.Total time: {total_time:.2f}s | Avg per text:{avg_time*1000:.2f}ms "
          f"| Speed: {len(texts)/total_time:.1f} texts/sec")
    return embeddings


if __name__ == "__main__":
    model = load_model()
    texts = []
    with open("data/processed/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            texts.append(chunk["text"])

    vectors = embed_texts(model, texts)
    print(f"\nTotal chunks embedded: {len(vectors)}")


