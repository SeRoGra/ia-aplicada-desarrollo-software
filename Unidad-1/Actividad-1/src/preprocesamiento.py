import json
import re
import unicodedata
from pathlib import Path

import spacy

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

NLP = spacy.load("es_core_news_sm")

SPANISH_STOPWORDS = NLP.Defaults.stop_words

def remove_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )

def basic_clean(text: str) -> str:
    # 1) minusculas
    text = text.lower().strip()
    # 2) quitar tildes
    text = remove_accents(text)
    # 3) remover símbolos raros (conservar letras/números/espacios)
    text = re.sub(r"[^a-z0-9\sñ]", " ", text)
    # 4) normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_remove_stopwords_and_lemmatize(text: str) -> str:
    doc = NLP(text)
    tokens = []
    for token in doc:
        if token.is_space:
            continue
        if token.text in SPANISH_STOPWORDS:
            continue
        if token.is_punct:
            continue
        lemma = token.lemma_.strip()
        if lemma and lemma not in SPANISH_STOPWORDS:
            tokens.append(lemma)
    return " ".join(tokens)

def preprocess_corpus(corpus: list[dict]) -> list[dict]:
    result = []
    for item in corpus:
        original = item["text"]
        cleaned = basic_clean(original)
        lemmatized = tokenize_remove_stopwords_and_lemmatize(cleaned)
        result.append({
            "id": item["id"],
            "category_hint": item.get("category_hint", ""),
            "text_original": original,
            "text_clean": cleaned,
            "text_lemmatized": lemmatized
        })
    return result

def main():
    corpus_path = DATA_DIR / "corpus_original.json"
    out_json = DATA_DIR / "corpus_limpio.json"

    # 1) Cargar corpus (ya sea real o simulado)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    # 2) Preprocesar
    processed = preprocess_corpus(corpus)

    # 3) Guardar limpio en JSON
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"OK -> generado: {out_json}")


if __name__ == "__main__":
    main()