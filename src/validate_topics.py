import json
from topic_modeling import fit_topic_model

def manually_label_topics(topic_keywords):
    labeled = {}
    for tid, keywords in topic_keywords.items():
        print(f"\nTopic {tid} keywords: {keywords}")
        label = input("  Give this topic a short name (e.g. 'Astronomy'): ").strip()
        quality = input("  Quality: good / unclear / bad? ").strip().lower()
        labeled[tid] = {"keywords": keywords, "label": label, "quality": quality}
    with open("topic_labels.json", "w", encoding="utf-8") as f:
        json.dump(labeled, f, indent=2)
    print("\nSaved to topic_labels.json")
    return labeled