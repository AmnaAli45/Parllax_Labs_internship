from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, compound

def evaluate_accuracy(labeled_examples):
    correct = 0
    mistakes = []
    for text, true_label in labeled_examples:
        predicted_label, score = get_sentiment(text)
        if predicted_label == true_label:
            correct += 1
        else:
            mistakes.append((text, true_label, predicted_label, score))
    accuracy = correct / len(labeled_examples) if labeled_examples else 0
    return accuracy, mistakes