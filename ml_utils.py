import re


def clean_text(text):
    """Same cleaning is used for training and for prediction."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()