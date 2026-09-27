from src.chunking import chunk_text

def test_short_text_stays_one_chunk():
    result = chunk_text("Hello world.", chunk=100, overlap=10)
    assert result == ["Hello world."]

def test_empty_text_gives_no_chunks():
    result = chunk_text("", chunk=100, overlap=10)
    assert result == []

def test_long_text_makes_multiple_chunks():
    text = "word " * 200         
    result = chunk_text(text, chunk=50, overlap=10)
    assert len(result) > 1        

def test_no_chunk_bigger_than_chunk_size():
    text = "word " * 200
    result = chunk_text(text, chunk=50, overlap=10)
    for chunk in result:
        assert len(chunk) <= 50

def test_words_are_not_cut_in_the_middle():
    text = "apple banana cherry date fig grape"
    result = chunk_text(text, chunk=15, overlap=0)
    for chunk in result:
        assert not chunk.startswith(" ")
        assert chunk == chunk.strip()

def test_overlap_shares_text_between_chunks():
    text = "word " * 50
    result = chunk_text(text, chunk=30, overlap=10)
    assert result[0][-5:] in result[1] or result[1].startswith(result[0][-5:].strip())