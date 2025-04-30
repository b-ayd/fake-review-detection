def clean_text(text):
    import re, string
    from bs4 import BeautifulSoup
    import pandas as pd

    if pd.isna(text): 
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()