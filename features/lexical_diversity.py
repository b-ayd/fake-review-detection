def compute_lexical_diversity(texts):
    """Compute lexical diversity (unique words / total words) for a list of texts."""
    diversities = []
    for text in texts:
        if not text or text.strip() == "":
            diversities.append(0.0)
            continue
        words = text.split()
        if len(words) == 0:
            diversities.append(0.0)
        else:
            diversities.append(len(set(words)) / len(words))
    return diversities