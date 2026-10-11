import json
import chromadb
from topic_modeling import fit_topic_model
from edge_cases import safe_assign_topic
from sentiment_analysis import get_sentiment
from embeddings import load_model, embed_texts
from vector_store import get_collection


def add_chunks_with_metadata(collection, ids, texts, embeddings, topics, sentiments):
    metadatas = []
    for topic, sentiment in zip(topics, sentiments):
        metadatas.append({
            "topic": str(topic),
            "sentiment": sentiment,
        })

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"{len(ids)} chunks metadata ke saath daale gaye.")


def filtered_search(collection, query_embedding, filters, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filters,
    )
    return results


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = []
    with open("data/processed/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks.")

    print("Training topic model (this can take a few minutes)...")
    vectorizer, lda, topic_keywords = fit_topic_model(texts, n_topics=10)
    print("Done training. Topics found:")
    for tid, keywords in topic_keywords.items():
        print(f"  Topic {tid}: {keywords}")

    print("\nAssigning topics and sentiment to each chunk...")
    topics = []
    sentiments = []
    for text in texts:
        topic_id, status = safe_assign_topic(text, vectorizer, lda)
        topics.append(topic_id if topic_id is not None else "unknown")

        sentiment_label, score = get_sentiment(text)
        sentiments.append(sentiment_label)
    print("Done assigning.")

    print("\nGenerating embeddings...")
    model = load_model()
    embeddings = embed_texts(model, texts)
    print("Done embedding.")

    print("\nIngesting into ChromaDB...")
    collection = get_collection()
    ids = [c["chunk_id"] for c in chunks]
    add_chunks_with_metadata(collection, ids, texts, embeddings, topics, sentiments)

    print("\n--- Demo: filtered search ---")
    query_vector = model.encode("technology and innovation").tolist()

    print("\nWithout filter:")
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [{meta['topic']} | {meta['sentiment']}] {doc[:60]}")

    print("\nWith filter sentiment=positive:")
    filtered = filtered_search(collection, query_vector, {"sentiment": "positive"}, top_k=3)
    for doc, meta in zip(filtered["documents"][0], filtered["metadatas"][0]):
        print(f"  [{meta['topic']} | {meta['sentiment']}] {doc[:60]}")