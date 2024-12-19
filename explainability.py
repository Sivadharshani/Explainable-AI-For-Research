from collections import Counter

def generate_explanation(text):
    # Simplified logic: Importance based on word frequency
    words = text.split()
    word_counts = Counter(words)
    max_count = max(word_counts.values())
    explanation_data = [{"word": word, "importance": count / max_count} for word, count in word_counts.items()]
    return explanation_data
