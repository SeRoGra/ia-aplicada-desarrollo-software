import json
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


# -----------------------------
# Normalización (tildes, minúsculas)
# -----------------------------
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


# -----------------------------
# Carga del diccionario
# -----------------------------
def load_dictionary() -> dict:
    path = DATA_DIR / "diccionario_legal.json"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el diccionario: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------
# Lectura de archivos (TXT / PDF)
# -----------------------------
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def read_pdf(path: Path) -> str:
    try:
        import PyPDF2
    except ImportError:
        raise ImportError(
            "Para leer PDF debes instalar PyPDF2.\n"
            "Ejecuta: pip install PyPDF2"
        )

    reader = PyPDF2.PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append((page.extract_text() or "").strip())
    return "\n".join([p for p in parts if p]).strip()


def load_input_from_file(path_str: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_txt(path)
    if suffix == ".pdf":
        return read_pdf(path)

    # fallback: tratar como texto
    return read_txt(path)


# -----------------------------
# Clasificación basada en reglas
# -----------------------------
def predict_category(text: str, dictionary: dict) -> tuple[str, dict]:
    text_norm = normalize_text(text)
    scores = {}

    for category, keywords in dictionary.items():
        # normalizamos keywords también por seguridad
        kws = [normalize_text(k) for k in keywords]
        score = sum(1 for kw in kws if kw and kw in text_norm)
        scores[category] = score

    best = max(scores, key=scores.get) if scores else "sin_categoria"
    if not scores or scores[best] == 0:
        return "sin_categoria", scores

    return best, scores


# -----------------------------
# Respuesta simulada (JustIA)
# -----------------------------
def simulated_response(question: str, category: str) -> str:
    base = (
        "=== JustIA – Respuesta (Simulación Académica) ===\n"
        f"Categoría identificada: {category}\n\n"
        "Orientación preliminar:\n"
    )

    if category == "familia":
        detail = "- El caso parece estar relacionado con derecho de familia (custodia, alimentos, visitas, etc.)."
    elif category == "laboral":
        detail = "- El caso parece estar relacionado con derecho laboral (contrato, salario, despido, prestaciones, etc.)."
    elif category == "penal":
        detail = "- El caso parece estar relacionado con derecho penal (delitos, capturas, fiscalía, audiencias, etc.)."
    elif category == "civil":
        detail = "- El caso parece estar relacionado con derecho civil (obligaciones, contratos, responsabilidad, daños, etc.)."
    elif category == "constitucional":
        detail = "- El caso parece estar relacionado con derecho constitucional (tutela, derechos fundamentales, protección, etc.)."
    else:
        detail = "- No se identificó una categoría con suficiente evidencia en el texto."

    closing = (
        "\n\nNota:\n"
        "Esta respuesta es una simulación con fines académicos y no reemplaza el análisis jurídico profesional."
    )

    return base + detail + closing


# -----------------------------
# Menú por consola
# -----------------------------
def print_menu():
    print("\n=== Menú – Actividad 3 (JustIA) ===")
    print("1) Ingresar pregunta legal (texto)")
    print("2) Cargar documento de entrada (.txt o .pdf)")
    print("3) Clasificar contenido cargado")
    print("4) Ver vista previa del contenido cargado")
    print("5) Salir")


def main():
    dictionary = load_dictionary()

    last_source = None   # "texto" | "archivo"
    last_text = ""

    while True:
        print_menu()
        opt = input("Seleccione una opción: ").strip()

        if opt == "1":
            q = input("\nIngrese su pregunta legal:\n> ").strip()
            if not q:
                print("No se ingresó texto.")
                continue
            last_source = "texto"
            last_text = q
            print("OK. Pregunta almacenada.")

        elif opt == "2":
            p = input("\nIngrese la ruta del archivo (.txt o .pdf):\n> ").strip()
            if not p:
                print("No se ingresó ruta.")
                continue
            try:
                content = load_input_from_file(p)
                if not content:
                    print("El archivo se leyó, pero no se extrajo contenido.")
                    continue
                last_source = "archivo"
                last_text = content
                print("OK. Documento cargado.")
            except Exception as e:
                print(f"Error al cargar el documento: {e}")

        elif opt == "3":
            if not last_text.strip():
                print("No hay contenido cargado. Use la opción 1 o 2.")
                continue

            category, scores = predict_category(last_text, dictionary)
            print("\n=== Resultado de clasificación ===")
            print(f"Fuente: {last_source}")
            print(f"Categoría estimada: {category}")
            print(f"Puntuaciones: {scores}\n")
            print(simulated_response(last_text, category))

        elif opt == "4":
            if not last_text.strip():
                print("No hay contenido cargado.")
                continue
            preview = last_text[:600].replace("\n", " ")
            print(f"\nFuente: {last_source}")
            print(f"Vista previa (600 chars): {preview}{'...' if len(last_text) > 600 else ''}")

        elif opt == "5":
            print("Saliendo del sistema. Gracias por utilizar JustIA.")
            break

        else:
            print("Opción inválida. Intente nuevamente.")


if __name__ == "__main__":
    main()