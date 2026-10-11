import json
from topic_modeling import fit_topic_model

def is_too_short(text, min_words=15):
    word_count = len(text.split())
    return word_count < min_words


def vocabulary_coverage(text, vectorizer):
    words = text.lower().split()
    if not words:
        return 0.0
    known_words = set(vectorizer.get_feature_names_out())
    recognized = sum(1 for w in words if w.strip(".,!?;:") in known_words)
    return recognized / len(words)

def safe_assign_topic(text, vectorizer, lda, min_words=15, min_coverage=0.15):
    if is_too_short(text, min_words):
        return None, "too_short"
    coverage = vocabulary_coverage(text, vectorizer)
    if coverage < min_coverage:
        return None, "low_vocabulary_coverage (possibly jargon-heavy)"
    doc_term_matrix = vectorizer.transform([text])
    topic_distribution = lda.transform(doc_term_matrix)
    topic_id = topic_distribution.argmax()
    return int(topic_id), "ok"

if __name__ == "__main__":

    texts = []
    with open("data/processed/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    print(f"Loaded {len(texts)} chunks.")
    print("Training topic model (this can take a few minutes)...")
    vectorizer, lda, topic_keywords = fit_topic_model(texts, n_topics=10)
    print("Done training.")

    status_count = {}
    for text in texts[:500]:   
        _, status = safe_assign_topic(text, vectorizer, lda)
        status_count[status] = status_count.get(status, 0) + 1

    print("\nEdge case summary (first 500 chunks):")
    print(status_count)