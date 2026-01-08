import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

def load_dictionary():
    path = DATA_DIR / "diccionario_legal.json"
    return json.loads(path.read_text(encoding="utf-8"))

def predict_category(text: str, dictionary: dict) -> tuple[str, dict]:
    text_lower = text.lower()
    scores = {}

    for category, keywords in dictionary.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score

    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "sin_categoria", scores

    return best_category, scores

def main():
    dictionary = load_dictionary()

    text = input("Ingrese el texto jurídico a clasificar:\n> ")
    category, scores = predict_category(text, dictionary)

    print("\nResultado de la clasificación:")
    print(f"Categoría estimada: {category}")
    print("Detalle de puntuaciones:", scores)

if __name__ == "__main__":
    main()