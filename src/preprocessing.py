import re
from langdetect import detect
import html
import unicodedata
from langdetect.lang_detect_exception import LangDetectException
import json
from pathlib import Path
import random

# HTML Stripping

def html_stripping(text):
    clean_text = re.sub(re.compile('<.*?>'),"",text)
    clean_text = re.sub(r"</?[a-zA-Z][^>]*>", "",clean_text)
    clean_text = re.sub(r"<!--.*?-->","",clean_text)#comments ko remove kre ga
    clean_text = html.unescape(clean_text) # escape characters ko remove kre ga
    return clean_text


# Unicode Normalization
def unicode_normalization(text):
    clean_text = unicodedata.normalize('NFKC',text)
    clean_text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", clean_text)
    return clean_text

# WhiteSpace Removal
def remove_whitespaces(text):
    clean_text = text.replace("\xa0"," ") # normal space
    clean_text = clean_text.replace("\r\n", "\n").replace("\r", "\n") #line endings -> \n
    clean_text = re.sub(r"[ \t]+", " ", clean_text) #tabs and sapces handling
    clean_text = re.sub(r" ?\n ?", "\n", clean_text)  # trim spaces around newlines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text) #max one blank line between paragraphs
    return clean_text.strip()  

# Language Filtering
def lang_filter(text):
    try:
        if detect(text[:1000]) == 'en':# phle 1000 characters ko check krna kafi hai 
            return text
    except LangDetectException:
        pass
    
    return ""


# Clean Text Pipeline
def clean_text_pipeline(text):
    if not isinstance(text,str):
        return ""
    text = html_stripping(text)
    text = unicode_normalization(text)
    text = remove_whitespaces(text)
    text = lang_filter(text)
    return text



# Storing lean dataset
def run(input_path, output_path, min_chars=200):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc["text"] = clean_text_pipeline(doc.get("text", ""))
            if len(doc["text"]) < min_chars: # empty ya non-English ya bohot chhota text ho to drop ho jaye ga 
                dropped += 1
                continue
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept={kept}, dropped={dropped}")
    



# ValidationS
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

def validate_corpus(path, min_docs=4000, lang_sample=200):
    problems, ids, texts, n = [], set(), [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            n += 1
            text = doc.get("text", "")
            texts.append(text)
            if not text.strip():
                problems.append(f"doc {n}: empty text")
            if TAG_RE.search(text):
                problems.append(f"doc {n}: HTML tag left")
            if re.search(r"&(amp|nbsp|lt|gt|quot|#\d+);", text):
                problems.append(f"doc {n}: HTML entity left")
            if re.search(r"[\u200b\ufeff\xa0]", text):
                problems.append(f"doc {n}: invisible char left")
            if re.search(r"[ \t]{2,}| \n|\n |\n{3,}", text) or text != text.strip():
                problems.append(f"doc {n}: whitespace not clean")
            if doc.get("id") in ids:
                problems.append(f"doc {n}: duplicate id")
            ids.add(doc.get("id"))

    random.seed(0)
    for text in random.sample(texts, min(lang_sample, len(texts))):
        if lang_filter(text) == "":
            problems.append("sampled doc is not English")
    if n < min_docs:
        problems.append(f"only {n} documents (need >= {min_docs})")

    print(f"validated {n} documents, {len(problems)} problem(s)")
    for p in problems[:10]:
        print("  -", p)
    return not problems


if __name__ == "__main__":
    RAW = "data/raw/wikipedia_5000.jsonl"          
    OUT = "data/processed/clean_corpus.jsonl"
    run(RAW,OUT)
    ok = validate_corpus(OUT)
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    
    