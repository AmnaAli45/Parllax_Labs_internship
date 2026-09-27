from src.embeddings import embed_texts


class FakeModel:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def test_embed_texts_returns_one_vector_per_text():
        texts = ["hello", "world", "foo"]
        result = embed_texts(FakeModel(), texts)
        assert len(result) == len(texts)    
        
    def test_embed_empty_list_returns_empty():
        result = embed_texts(FakeModel(), [])
        assert result == []

    def test_each_embedding_has_same_length():
        texts = ["a", "bb", "ccc"]
        result = embed_texts(FakeModel(), texts)
        lengths = [len(vec) for vec in result]
        assert lengths == [3,3,3]    

    def test_embed_texts_does_not_crash_on_single_text():
        result = embed_texts(FakeModel(), ["just one sentence"])
        assert len(result) == 1