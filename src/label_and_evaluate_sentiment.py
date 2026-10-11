import csv
import json
import random
from pathlib import Path

from sentiment_analysis import evaluate_accuracy

CHUNKS_PATH = "data/processed/chunks.jsonl"
CSV_PATH = "sentiment_labeling.csv"
SAMPLE_SIZE = 20

def create_labeling_csv():
    random.seed(42)
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    sample = random.sample(chunks, min(SAMPLE_SIZE, len(chunks)))
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "text", "true_label"])
        for chunk in sample:
            writer.writerow([chunk.get("chunk_id", ""), chunk["text"][:300], ""])
    print(f"Wrote {len(sample)} chunks to {CSV_PATH}.")
    print("Open it in Excel, fill the 'true_label' column with")
    print("positive, negative, or neutral for each row, save it,")
    print("then run this script again to see the accuracy.")

def run_evaluation():
    labeled_examples = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["true_label"].strip():
                labeled_examples.append((row["text"], row["true_label"].strip()))
    if not labeled_examples:
        print(f"No labels found in {CSV_PATH} yet. Fill in the 'true_label' column first.")
        return
    accuracy, mistakes = evaluate_accuracy(labeled_examples)
    print(f"Accuracy: {accuracy:.0%} ({len(labeled_examples) - len(mistakes)}/{len(labeled_examples)} correct)\n")
    if mistakes:
        print("Mistakes:")
        for text, true_label, predicted_label, score in mistakes:
            print(f"  Text: {text}")
            print(f"  True: {true_label} | Predicted: {predicted_label} (score {score:+.2f})\n")

if __name__ == "__main__":
    if Path(CSV_PATH).exists():
        run_evaluation()
    else:
        create_labeling_csv()