from transformers import pipeline
from typing import Dict, Any

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes the sentiment of a given text and returns the sentiment label
    (positive or negative) and a probability score.
    """
    result = sentiment_analyzer(text)
    
    # The result is a list containing one dictionary
    if isinstance(result, list) and result:
        res_dict = result[0]
        label = res_dict.get('label')
        score = res_dict.get('score')
    else:
        # Handle cases where the result is not as expected
        return {"sentiment": "unknown", "probability": 0.0}

    # The model might return "POSITIVE" or "NEGATIVE". We will standardize it.
    if label == "POSITIVE":
        sentiment = "positive"
    elif label == "NEGATIVE":
        sentiment = "negative"
    else:
        # Fallback for other labels, though typical models use POSITIVE/NEGATIVE
        sentiment = str(label).lower() if label is not None else "unknown"

    return {"sentiment": sentiment, "probability": score}

if __name__ == "__main__":
    sample_text = "I love using this new library! It's so easy and intuitive."
    analysis = analyze_sentiment(sample_text)
    print(f"Sentiment analysis for: '{sample_text}'")
    print(f"Sentiment: {analysis['sentiment']}")
    print(f"Probability: {analysis['probability']:.4f}")

    sample_text_2 = "I am not happy with the customer service."
    analysis_2 = analyze_sentiment(sample_text_2)
    print(f"\nSentiment analysis for: '{sample_text_2}'")
    print(f"Sentiment: {analysis_2['sentiment']}")
    print(f"Probability: {analysis_2['probability']:.4f}") 