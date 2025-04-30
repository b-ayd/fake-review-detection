import textstat

def extract_readability_features(texts):
    """Extract readability scores from a list of texts."""
    flesch_reading_ease = []
    flesch_kincaid_grade = []
    avg_sentence_length = []
    for text in texts:
        if not text or text.strip() == "":
            flesch_reading_ease.append(0)
            flesch_kincaid_grade.append(0)
            avg_sentence_length.append(0)
            continue
        flesch_reading_ease.append(textstat.flesch_reading_ease(text))
        flesch_kincaid_grade.append(textstat.flesch_kincaid_grade(text))
        avg_sentence_length.append(textstat.avg_sentence_length(text))
    
    return flesch_reading_ease, flesch_kincaid_grade, avg_sentence_length