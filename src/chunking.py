import json
from pathlib import Path

def chunk_text(text,chunk=300,overlap=50):
    chunks = []
    start = 0
    len_text = len(text)
    
    while (start<len_text): # tbb tk loop chale gy jbb tk text end tk na pohanch jaye
        end = start + chunk
        if end >= len_text:
            chunked = text[start:len_text].strip() # agr end prr aa gaye hain to jo hai bss us ko chunk bna do
            if chunked:
                chunks.append(chunked)
                break
        
        while end > start and text[end] != " ":
            end -=1 # word ko darmyan se cut nhi krna is lye peeche ale space tk jana hai 
        if end == start : # peeche koi bhi space nhi mila (aik hi lamba word hai)
            end = start + chunk
        
        chunked = text[start:end].strip()
        if chunked:
            chunks.append(chunked)
        
        start = end - overlap # next chunk peeche wale se overlap krna hai 
    return chunks

def chunk_dataset(input_path, output_path, chunk_size=300, overlap=50):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total_docs = 0
    total_chunks = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            total_docs += 1

            pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)

            for i, piece in enumerate(pieces):
                chunk_record = {
                    "chunk_id": f"{doc['id']}_chunk{i}",
                    "doc_id": doc["id"],
                    "title": doc.get("title", ""),
                    "text": piece,
                }
                fout.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"documents={total_docs}, chunks={total_chunks}")


if __name__ == "__main__":
    RAW = "data/processed/clean_corpus.jsonl"      
    OUT = "data/processed/chunks.jsonl"           
    chunk_dataset(RAW, OUT, chunk_size=300, overlap=50)
        