from pyvi import ViTokenizer

def preprocess_text(text: str, use_pyvi: bool = True):
    text = text.strip()
    if not use_pyvi:
        return text

    return ViTokenizer.tokenize(text)
