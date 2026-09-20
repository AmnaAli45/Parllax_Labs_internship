# RAG Data Pipeline: Week 1 (Parallax Labs Internship)

A reproducible foundation for a Retrieval-Augmented Generation (RAG) system: a **verified Python environment**, a **modular text-cleaning pipeline with unit tests**, and a **validated clean corpus** built from real Wikipedia articles.

---

## 1. What I built

| Deliverable | Where |
|---|---|
| Pinned, reproducible environment + verification script | `requirements.txt`, `scripts/verify_environment.py` |
| Real-world dataset acquisition (Wikipedia, 5,000+ documents) | `scripts/download_data.py` → `data/raw/` |
| Modular cleaning pipeline (HTML, Unicode, whitespace, language) | `src/preprocessing.py` |
| Unit tests for every cleaning function | `tests/test_preprocessing.py` |
| Corpus builder + automatic validation | `run()` and `validate_corpus()` in `src/preprocessing.py` |
| Validated clean dataset | `data/processed/clean_corpus.jsonl` (git-ignored, see Section 8) |

---

## 2. Repository structure

```
.
├── README.md
├── requirements.txt              # pinned dependencies
├── .gitignore                    # ignores venv/, .env, data/
├── scripts/
│   ├── verify_environment.py     # checks the environment works
│   └── download_data.py          # downloads raw Wikipedia articles
├── src/
│   ├── __init__.py
│   └── preprocessing.py          # cleaning functions + corpus builder + validation
├── tests/
│   └── test_preprocessing.py     # unit tests
├── samples/
│   └── clean_corpus_sample.jsonl # 20-document sample of the cleaned output
└── data/                         # NOT committed (git-ignored)
    ├── raw/                      # raw downloaded data
    └── processed/                # cleaned corpus
```

---

## 3. Approach

### 3.1 Environment
All dependencies are **pinned to exact versions** in `requirements.txt` so every machine installs the same libraries: `sentence-transformers`, `torch`, `transformers`, `spacy`, `pandas`, `numpy`, `chromadb`, plus `langdetect` (language filtering), `datasets` (data download) and `pytest` (tests).
`scripts/verify_environment.py` proves the environment works: it compares installed versions against `requirements.txt` and runs a small real operation with each library (a torch tensor sum, a spaCy tokenizer, an in-memory ChromaDB insert + query, and so on).

### 3.2 Dataset
* **Source:** English Wikipedia (`wikimedia/wikipedia`, snapshot `20231101.en`) through Hugging Face `datasets`.
* **Size:** <RAW_COUNT> articles downloaded, so that at least 5,000 remain after cleaning.
* **Sampling:** streamed and shuffled with a fixed seed (`42`), so the sample is random rather than all "A" titles, and reproducible.
* **Stored raw and untouched** in `data/raw/`. Cleaning never modifies the raw files.

### 3.3 Cleaning pipeline
Each step is a small independent function in `src/preprocessing.py`, chained by `clean_text_pipeline()`:

| Order | Function | What it does | Why |
|---|---|---|---|
| 1 | `html_stripping` | Removes `<script>`/`<style>` blocks and their content, turns block tags (`<p>`, `<br>`, `<h1>`...) into line breaks, removes other tags, decodes entities (`&amp;`, `&nbsp;`) | Markup is noise for embeddings; decoding first can produce odd characters (e.g. non-breaking spaces) that later steps clean up |
| 2 | `unicode_normalization` | NFKC normalization, removes zero-width characters and soft hyphens | Makes visually identical text byte-identical (ligatures like "ﬁ" → "fi", full-width letters, composed vs. decomposed accents) |
| 3 | `remove_whitespaces` | Converts non-breaking spaces and line endings, collapses runs of spaces/tabs, trims spaces around newlines, allows at most one blank line, strips the ends | Keeps paragraph structure (useful for chunking later) but removes layout noise |
| 4 | `lang_filter` | Keeps text only if it is detected as English (first 1,000 characters, fixed seed) | The knowledge base should be one language; `langdetect` is seeded so results are deterministic |

**Order matters:** HTML → Unicode → whitespace → language, so every step sees the output of the one before it, and language detection runs on clean text.

**Document-level rule:** `run()` also drops documents that are empty, non-English or shorter than 200 characters after cleaning (redirects and stubs carry no useful knowledge).

**Robustness:** `lang_filter` catches `langdetect` exceptions (empty text, numbers only), and `clean_text_pipeline` returns `""` for non-string input instead of crashing.

### 3.4 Validation
After cleaning, `validate_corpus()` re-reads the output file and checks that:
* no document has empty text,
* no HTML tags or HTML entities remain,
* no invisible characters / non-breaking spaces remain,
* whitespace is clean (no double spaces, no 3+ newlines, no leading/trailing space),
* document ids are unique,
* a random sample of 200 documents is detected as English,
* the corpus contains at least 5,000 documents.

The script prints `VALIDATION PASSED` or lists the problems found.

### 3.5 Testing
`tests/test_preprocessing.py` contains focused unit tests, each checking one behaviour:
* **HTML:** tags removed, entities decoded
* **Unicode:** ligatures normalized, zero-width characters removed
* **Whitespace:** extra spaces, extra blank lines, leading/trailing spaces
* **Language:** English kept, French dropped, empty text does not crash
* **Pipeline:** end-to-end cleaning of a messy HTML string

---

## 4. Setup

Requires **Python 3.10+** (developed on <YOUR_PYTHON_VERSION>).

```powershell
# 1. Clone
git clone https://github.com/AmnaAli45/Parllax_Labs_internship.git
cd Parllax_Labs_internship

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1            # Windows PowerShell
# source venv/bin/activate           # macOS / Linux

# 3. Install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Install the spaCy English model
python -m spacy download en_core_web_sm
```

> If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
> Note: `torch` is large (hundreds of MB on Linux/Windows), so the first install can take a few minutes.

---

## 5. How to run

**Always run commands from the project root.**

```powershell
# 1. Verify the environment
python scripts/verify_environment.py
#    optional: also download a small embedding model and encode text
python scripts/verify_environment.py --full

# 2. Download the raw dataset  ->  data/raw/wikipedia_6500.jsonl
python scripts/download_data.py

# 3. Run the unit tests
python -m pytest -v

# 4. Clean the data and validate  ->  data/processed/clean_corpus.jsonl
python src/preprocessing.py
```

Using the cleaning functions in your own code:

```python
from src.preprocessing import clean_text_pipeline

clean_text_pipeline("<p>Hello&nbsp;   world, this is a short English sentence.</p>")
# returns cleaned text, or "" if the text is not English
```

---

## 6. Results

**Environment verification**
```
<PASTE THE OUTPUT OF: python scripts/verify_environment.py>
```

**Unit tests**
```
<PASTE THE OUTPUT OF: python -m pytest -v>
```

**Corpus cleaning and validation**
```
<PASTE THE OUTPUT OF: python src/preprocessing.py>
```

| Metric | Value |
|---|---|
| Raw documents | <RAW_COUNT> |
| Documents kept | <KEPT> |
| Documents dropped (empty / non-English / < 200 chars) | <DROPPED> |
| Validation | PASSED |

---

## 7. Design decisions and limitations

* **`langdetect` is statistical.** It can misjudge very short or mixed-language text, so the pipeline only uses the first 1,000 characters and validation checks a random sample rather than claiming 100% accuracy.
* **NFKC normalization** is lossy on purpose (e.g. superscripts and ligatures are flattened), which is good for search and embeddings but not suitable if the original typography must be preserved.

* **Only English** is kept in this version. The target language is a single argument in `lang_filter`, so extending it is straightforward.


---

## 8. About the data files

Raw and processed data are **git-ignored** (`data/`) because they are large and reproducible. To regenerate them, run steps 2 and 4 in Section 5. A 20-document preview of the cleaned output is committed in `samples/clean_corpus_sample.jsonl`.

---

## 9. Tech stack

Python · sentence-transformers · PyTorch · Transformers · spaCy · pandas · NumPy · ChromaDB · langdetect · Hugging Face `datasets` · pytest

## 10. Author

Amna Ali: Parallax Labs Internship