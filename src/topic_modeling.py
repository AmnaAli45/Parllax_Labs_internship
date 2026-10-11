from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def fit_topic_model(texts, n_topics=5, n_top_words=8, min_df=2, max_df=0.9):
    vectorizer = CountVectorizer(max_df=max_df,min_df=min_df,stop_words="english",)
    doc_term_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics,random_state=42,max_iter=50,)
    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for topic_id, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        words = [feature_names[i] for i in top_indices]
        topic_keywords[topic_id] = ", ".join(words)

    return vectorizer, lda, topic_keywords


def assign_topics(texts, vectorizer, lda):
  
    doc_term_matrix = vectorizer.transform(texts)
    topic_distribution = lda.transform(doc_term_matrix)
    return topic_distribution.argmax(axis=1).tolist()