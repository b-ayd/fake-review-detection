from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np

# Make sure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Initialize once
sia = SentimentIntensityAnalyzer()

def extract_sentiment_scores(texts):
    """Extract compound sentiment scores from a list of texts."""
    sentiment_scores = []
    for text in texts:
        if not text or text.strip() == "":
            sentiment_scores.append(0.0)
            continue
        scores = sia.polarity_scores(text)
        sentiment_scores.append(scores["compound"])  # compound = overall sentiment
    return sentiment_scores

def compute_sentiment_rating_inconsistency(texts, ratings):
    """
    Computes inconsistency between sentiment and rating.
    
    Args:
        texts (list): List of review texts.
        ratings (list): Corresponding ratings (1-5).
        
    Returns:
        np.ndarray: Inconsistency scores (absolute difference).
    """
    sentiment_scores = extract_sentiment_scores(texts)
    
    # Normalize ratings from 1-5 to [-1, 1]
    ratings_normalized = [(2 * (r - 1) / 4) - 1 for r in ratings]
    
    # Compute inconsistency
    inconsistencies = [abs(sent - rating) for sent, rating in zip(sentiment_scores, ratings_normalized)]
    
    return np.array(inconsistencies).reshape(-1, 1)


