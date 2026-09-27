import chromadb


def get_collection(db_path="chroma_db", collection_name="my_chunks"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def add_chunks(collection, ids, texts, embeddings, metadatas=None):
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

def semantic_search(collection, query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results   