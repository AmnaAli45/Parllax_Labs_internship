
DISTANCE_THRESHOLD = 1.0
REFUSAL_PHRASES = [
    "i don't have enough information",
    "i do not have enough information",
    "cannot answer",
    "can't answer",
    "not mentioned in the context",
]
def is_in_domain(distances, threshold=DISTANCE_THRESHOLD):
    if not distances:
        return False
    best_distance = min(distances)
    return best_distance <= threshold

def is_refusal(answer_text):
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


OUT_OF_DOMAIN_MESSAGE = (
    "This question does not seem to be covered by the available documents, "
    "so I can't answer it reliably."
)