# RAG Data Pipeline (Parallax Labs Internship)

A reproducible foundation for a Retrieval Augmented Generation (RAG) system. This document covers a verified Python environment, a modular text cleaning pipeline with unit tests, a validated clean corpus built from real Wikipedia articles, a chunking strategy, and how ChromaDB edge cases are handled.

---

## 1. What I built

| Deliverable | Where |
|---|---|
| Pinned, reproducible environment and verification script | `requirements.txt`, `scripts/verify_environment.py` |
| Real world dataset acquisition (Wikipedia, 5,000+ documents) | `scripts/download_data.py` to `data/raw/` |
| Modular cleaning pipeline (HTML, Unicode, whitespace, language) | `src/preprocessing.py` |
| Unit tests for every cleaning function | `tests/test_preprocessing.py` |
| Corpus builder and automatic validation | `run()` and `validate_corpus()` in `src/preprocessing.py` |
| Validated clean dataset | `data/processed/clean_corpus.jsonl` (git ignored, see Section 8) |
| Fixed size chunking of the clean corpus | `src/chunking.py` |
| Unit tests for chunking | `tests/test_chunking.py` |
| Sentence embeddings with timing logs | `src/embeddings.py` |
| ChromaDB setup, ingestion, and semantic search | `src/vector_store.py` |
| Retrieval performance and latency testing script | `scripts/test_retrieval.py` |
| Chunking strategy and ChromaDB edge case documentation | This document, Sections 10 and 11 |

---

## 2. Repository structure

```
.
├── README.md
├── requirements.txt              (pinned dependencies)
├── .gitignore                    (ignores venv/, .env, data/)
├── scripts/
│   ├── verify_environment.py     (checks the environment works)
│   ├── download_data.py          (downloads raw Wikipedia articles)
│   └── test_retrieval.py         (tests retrieval latency for several queries)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          (cleaning functions, corpus builder, validation)
│   ├── chunking.py               (text chunking)
│   ├── embeddings.py             (sentence embedding generation, timing)
│   └── vector_store.py           (ChromaDB setup, ingestion, semantic search)
├── tests/
│   ├── test_preprocessing.py     (unit tests for cleaning)
│   └── test_chunking.py          (unit tests for chunking)
├── samples/
│   └── clean_corpus_sample.jsonl (20 document sample of the cleaned output)
└── data/                         (not committed, git ignored)
    ├── raw/                      (raw downloaded data)
    └── processed/                (cleaned corpus, chunks)
```

---

## 3. Approach: environment

All dependencies are pinned to exact versions in `requirements.txt` so every machine installs the same libraries: `sentence-transformers`, `torch`, `transformers`, `spacy`, `pandas`, `numpy`, `chromadb`, plus `langdetect` (language filtering), `datasets` (data download), and `pytest` (tests).

`scripts/verify_environment.py` proves the environment works. It compares installed versions against `requirements.txt` and runs a small real operation with each library (a torch tensor sum, a spaCy tokenizer, an in memory ChromaDB insert and query, and so on).

---

## 4. Approach: dataset

* Source: English Wikipedia (`wikimedia/wikipedia`, snapshot `20231101.en`) through Hugging Face `datasets`.
* Size: `<RAW_COUNT>` articles downloaded, so that at least 5,000 remain after cleaning.
* Sampling: streamed and shuffled with a fixed seed (42), so the sample is random rather than all articles starting with the letter A, and the result is reproducible.
* Stored raw and untouched in `data/raw/`. Cleaning never modifies the raw files.

---

## 5. Approach: cleaning pipeline

Each step is a small independent function in `src/preprocessing.py`, chained by `clean_text_pipeline()`.

| Order | Function | What it does | Why |
|---|---|---|---|
| 1 | `html_stripping` | Removes `<script>` and `<style>` blocks and their content, turns block tags such as `<p>`, `<br>`, `<h1>` into line breaks, removes other tags, decodes entities such as `&amp;` and `&nbsp;` | Markup is noise for embeddings. Decoding first can produce unusual characters such as non breaking spaces, which the later steps clean up |
| 2 | `unicode_normalization` | NFKC normalization, removes zero width characters and soft hyphens | Makes visually identical text byte identical, for example ligatures such as "fi" written as one character become two separate letters, full width letters become normal letters, and composed accents match decomposed accents |
| 3 | `remove_whitespaces` | Converts non breaking spaces and line endings, collapses runs of spaces and tabs, trims spaces around newlines, allows at most one blank line, strips the ends | Keeps paragraph structure, which is useful for chunking later, while removing layout noise |
| 4 | `lang_filter` | Keeps text only if it is detected as English, checking the first 1,000 characters with a fixed seed | The knowledge base should be one language. `langdetect` is seeded so results are deterministic |

Order matters. HTML, then Unicode, then whitespace, then language, so every step sees the output of the one before it, and language detection runs on already clean text.

Document level rule: `run()` also drops documents that are empty, non English, or shorter than 200 characters after cleaning, since redirects and stub pages carry little useful knowledge.

Robustness: `lang_filter` catches `langdetect` exceptions for empty text or numbers only text, and `clean_text_pipeline` returns an empty string for non string input instead of crashing.

---

## 6. Approach: validation

After cleaning, `validate_corpus()` rereads the output file and checks that:

* no document has empty text,
* no HTML tags or HTML entities remain,
* no invisible characters or non breaking spaces remain,
* whitespace is clean, meaning no double spaces, no three or more consecutive newlines, and no leading or trailing space,
* document ids are unique,
* a random sample of 200 documents is detected as English,
* the corpus contains at least 5,000 documents.

The script prints `VALIDATION PASSED` or lists the problems found.

---

## 7. Approach: testing the cleaning pipeline

`tests/test_preprocessing.py` contains focused unit tests, each checking one behaviour:

* HTML: tags removed, entities decoded.
* Unicode: ligatures normalized, zero width characters removed.
* Whitespace: extra spaces, extra blank lines, leading and trailing spaces.
* Language: English kept, French dropped, empty text does not crash.
* Pipeline: end to end cleaning of a messy HTML string.

---

## 8. Approach: chunking strategy

### Why chunk at all

Embedding models can only turn a limited amount of text into one vector, and one vector for a whole article would blur many different facts together, making search less accurate. So each document is split into smaller pieces, called chunks, and each chunk gets its own embedding.

### The approach: fixed size splitting with a word safe cut and overlap

```python
def chunk_text(text, chunk=300, overlap=50):
    ...
```

1. Cut every `chunk` characters, 300 by default. This is simple and works the same way on any text, regardless of language or formatting.
2. Do not cut in the middle of a word. Before finalizing a cut point, the function searches backward for the nearest space, so a chunk always ends at a word boundary.
3. Guarantee forward progress. If the nearest space is very close to the start, within `overlap` characters, searching for a clean word boundary would produce a tiny chunk and could even make the next chunk start before the current one, causing an infinite loop. See the bug description below. In that case the function falls back to a hard cut at exactly `chunk` characters instead of searching for a space.
4. Overlap consecutive chunks by `overlap` characters, 50 by default. Each new chunk starts `overlap` characters before the previous one ended, so a sentence or fact that falls right on a chunk boundary is not cut away from its context in either chunk.

### Parameters chosen and why

| Parameter | Value | Reason |
|---|---|---|
| `chunk` | 300 characters | Small enough that each chunk holds about one paragraph of content, which keeps embeddings focused, and large enough to keep processing fast |
| `overlap` | 50 characters | Roughly 15 to 20 percent of the chunk size, enough to preserve context across a boundary without duplicating too much text |

### Known limitation

The hard cut fallback described in step 3 can, in rare cases, still cut a word in half, specifically when a very long run of text has no spaces at all, such as a long URL or a data blob. This trade off was chosen deliberately. It guarantees the algorithm always finishes in a fixed amount of time, rather than trying to preserve a word boundary at the cost of runtime safety.

### Bug found and fixed during development

An earlier version only checked whether `end` equals `start` to detect that no space was found. In practice, a space could be found very close to `start`, though not exactly at it, producing a tiny chunk. Because the next chunk's start position is `end` minus `overlap`, a tiny `end` made the next start smaller than the previous one, so the loop never finished. The fix replaced the check with a condition that compares `end` minus `start` against `overlap`. This catches both "no space found" and "space found too close" in one condition, and always guarantees the next chunk starts further ahead than the current one.

---

## 9. Approach: embeddings

`src/embeddings.py` uses `sentence-transformers` to turn each chunk's text into a numeric vector, and logs how long embedding takes.

* `load_model()` loads the model once and logs the loading time separately from the embedding time.
* `embed_texts()` processes texts in batches, using Python's `time` module to measure the total time and the average time per text, and prints a texts per second rate.

Timing is logged separately for loading and for embedding because loading only happens once per run, while embedding time scales with the number of chunks, so mixing the two numbers would be misleading.

---

## 10. Approach: ChromaDB setup, ingestion, and semantic search

`src/vector_store.py` wraps ChromaDB so the rest of the pipeline does not have to handle low level details directly.

* `get_collection()` opens a ChromaDB database on disk, creating it if it does not already exist, and returns a collection, which can be thought of as one table for chunks and their embeddings.
* `add_chunks()` ingests chunk ids, texts, and embeddings into the collection.
* `semantic_search()` embeds a query and returns the most similar chunks, ranked by distance, where a smaller distance means a closer match.

### ChromaDB edge cases

| Edge case | What ChromaDB does by default | How it is handled here |
|---|---|---|
| Adding an empty list of chunks | Raises a ValueError stating embeddings must be a non empty list | `add_chunks` checks the length first and prints a warning instead of calling ChromaDB |
| Ids, texts, and embeddings of different lengths | Raises a ValueError with a technical message | Checked explicitly first, with a clear message naming each length |
| Adding the same id twice in one call | Raises a DuplicateIDError | Not applicable here, since the pipeline generates unique ids combining the document id and the chunk index |
| Adding the same id twice across separate calls, for example when the ingestion script is run again | Silently does nothing. The original text and embedding are kept, with no error and no update | `add_chunks` uses `collection.upsert()` instead of `collection.add()`, so running ingestion again updates existing chunks instead of silently keeping stale data |
| Searching an empty collection | Returns an empty result with no crash | `semantic_search` checks `collection.count()` first and returns an empty result with a warning, without calling ChromaDB |
| Requesting more results than the number of items in the collection | ChromaDB automatically limits the result to however many items exist, with no error | Left as is, since ChromaDB already handles it safely. The requested number is also clamped using the smaller of the requested value and the total chunk count, for clarity |
| Query embedding dimension does not match the stored embeddings' dimension | Raises an InvalidArgumentError with a low level message | Caught and reraised as a ValueError with a plain language explanation of the likely cause |
| Collection name too short or invalid | Raises an InvalidArgumentError, since names must be between 3 and 512 characters, using only letters, digits, periods, underscores, or hyphens | Not handled in code. Use a descriptive collection name of at least 3 characters, for example `wikipedia_chunks` rather than a single letter |

### Why upsert instead of add

This was the most important fix. Ingestion scripts get run again often, for example after fixing a bug in cleaning, after downloading more data, or after changing the chunk size. With plain `add()`, running the script again on already ingested ids would leave the old, possibly wrong, chunks in the database forever, with no error to warn about it. `upsert()` makes running the ingestion script again safe. It updates existing chunks in place and adds any new ones.

---

## 11. Approach: retrieval performance and latency testing

`scripts/test_retrieval.py` runs a list of test queries against the collection and measures how long each one takes.

* Each query is timed from the moment it is embedded to the moment ChromaDB returns results, using Python's `time` module, and the result is reported in milliseconds.
* Each query's latency, top result, and distance are printed individually, so slow or poorly matched queries are easy to spot.
* A summary reports the average, minimum, and maximum latency across all test queries.
* The first query run in a session is often slower than the rest, since the model performs some one time setup work, so this is expected and not a sign of a problem.

---

## 12. Setup

Requires Python 3.10 or newer (developed on `<YOUR_PYTHON_VERSION>`).

```powershell
# 1. Clone
git clone https://github.com/AmnaAli45/Parllax_Labs_internship.git
cd Parllax_Labs_internship

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1            # Windows PowerShell
# source venv/bin/activate           # macOS or Linux

# 3. Install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Install the spaCy English model
python -m spacy download en_core_web_sm
```

If PowerShell blocks activation, run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Note that `torch` is large, several hundred megabytes on Linux and Windows, so the first install can take a few minutes.

---

## 13. How to run

Always run commands from the project root.

```powershell
# 1. Verify the environment
python scripts/verify_environment.py
#    optional: also download a small embedding model and encode text
python scripts/verify_environment.py --full

# 2. Download the raw dataset, saved to data/raw/wikipedia_6500.jsonl
python scripts/download_data.py

# 3. Run the unit tests
python -m pytest -v

# 4. Clean the data and validate, saved to data/processed/clean_corpus.jsonl
python src/preprocessing.py

# 5. Chunk the cleaned corpus, saved to data/processed/chunks.jsonl
python src/chunking.py

# 6. Generate embeddings and log timing
python src/embeddings.py

# 7. Set up ChromaDB and ingest chunks with embeddings
python src/vector_store.py

# 8. Test retrieval performance and log latency for several queries
python scripts/test_retrieval.py
```

Using the cleaning functions in your own code:

```python
from src.preprocessing import clean_text_pipeline

clean_text_pipeline("<p>Hello&nbsp;   world, this is a short English sentence.</p>")
# returns cleaned text, or an empty string if the text is not English
```

---

## 14. Results

Environment verification:
```
<PASTE THE OUTPUT OF: python scripts/verify_environment.py>
```

Unit tests:
```
<PASTE THE OUTPUT OF: python -m pytest -v>
```

Corpus cleaning and validation:
```
<PASTE THE OUTPUT OF: python src/preprocessing.py>
```

Chunking:
```
<PASTE THE OUTPUT OF: python src/chunking.py>
```

Embedding timing:
```
<PASTE THE OUTPUT OF: python src/embeddings.py>
```

Retrieval latency:
```
<PASTE THE OUTPUT OF: python scripts/test_retrieval.py>
```

| Metric | Value |
|---|---|
| Raw documents | `<RAW_COUNT>` |
| Documents kept after cleaning | `<KEPT>` |
| Documents dropped (empty, non English, or under 200 characters) | `<DROPPED>` |
| Validation | PASSED |
| Total chunks created | `<CHUNK_COUNT>` |
| Average embedding speed | `<TEXTS_PER_SEC>` texts per second |
| Average query latency | `<AVG_LATENCY_MS>` ms |

---

## 15. Design decisions and limitations

* `langdetect` is statistical. It can misjudge very short or mixed language text, so the pipeline only uses the first 1,000 characters, and validation checks a random sample rather than claiming complete accuracy.
* NFKC normalization is lossy on purpose, for example superscripts and ligatures are flattened, which is good for search and embeddings but not suitable if the original typography must be preserved.
* Only English is kept in this version. The target language is a single argument in `lang_filter`, so extending it is straightforward.
* Fixed size chunking with a word safe cut is simple and predictable, though it does not understand sentence or paragraph structure the way a recursive splitter would. This trade off favours reliability and speed over perfectly natural chunk boundaries.
* `upsert()` makes ingestion idempotent, meaning the script can be run again safely, at the cost of a small amount of extra write overhead compared to plain `add()`.

---

## 16. About the data files

Raw and processed data are git ignored, under `data/`, because they are large and reproducible. To regenerate them, run steps 2, 4, and 5 in Section 13. A 20 document preview of the cleaned output is committed in `samples/clean_corpus_sample.jsonl`.

---

## 17. Tech stack

Python, sentence-transformers, PyTorch, Transformers, spaCy, pandas, NumPy, ChromaDB, langdetect, Hugging Face `datasets`, pytest.

## 18. Author

Amna Ali, Parallax Labs Internship.