
import json
from pathlib import Path
from datasets import load_dataset

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=10_000)   # ta k A titles k ilawa bhi data rotate ho kr aye 

with open(out / "wikipedia_5000.jsonl", "w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        if i >= 5000:
            break
        f.write(json.dumps({
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "text": row["text"],
        }, ensure_ascii=False) + "\n")

print("Saved 5000 documents")