# =========================================
# Actividad 2: NER jurídico con spaCy (es_core_news_md)
# - EntityRuler: normas jurídicas y tipos de violencia
# - Matcher: fechas en formato dd/mm/yyyy
# - Exporta evidencias: HTML resaltado + TXT de resultados
# =========================================

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import spacy
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher
from spacy.tokens import Span


# -----------------------------
# 1) Textos base (10)
# -----------------------------
TEXTOS: List[str] = [
    "Según la Ley 100 de 1993, se regula el sistema de seguridad social integral en Colombia.",
    "La víctima reportó violencia intrafamiliar ocurrida el 12 de mayo de 2022 en Bogotá.",
    "Se investigan hechos por violencia económica y violencia patrimonial en el entorno familiar.",
    "El ciudadano Juan Pérez presentó recurso el 05/01/2021 ante el Juzgado 3 Civil del Circuito.",
    "Con fundamento en la Ley 1098 de 2006 se garantiza el interés superior del menor.",
    "Se solicitó medida de protección por violencia intrafamiliar y se fijó audiencia el 20 de agosto de 2023.",
    "La resolución administrativa indica la obligación según Ley 1562 de 2012 y normas concordantes.",
    "María Gómez afirmó que el hecho ocurrió el 02/02/2020 en Cali, Valle del Cauca.",
    "En aplicación de la Ley 640 de 2001 se ordena agotar conciliación antes de continuar el proceso.",
    "El proceso se adelanta ante la jurisdicción laboral ordinaria en Medellín y se solicita reintegro."
]


# -----------------------------
# 2) Configurar pipeline spaCy
# -----------------------------
def build_nlp() -> spacy.language.Language:
    nlp = spacy.load("es_core_news_md")

    # Insertar EntityRuler antes del NER para que nuestras reglas tengan prioridad
    ruler: EntityRuler = nlp.add_pipe("entity_ruler", before="ner")

    patrones = [
        # NORMA: "Ley 100 de 1993" / "Ley 1562 de 2012"
        {"label": "NORMA", "pattern": [{"LOWER": "ley"}, {"IS_DIGIT": True}, {"LOWER": "de"}, {"IS_DIGIT": True}]},

        # VIOLENCIA: tipos relevantes del caso
        {"label": "VIOLENCIA", "pattern": "violencia intrafamiliar"},
        {"label": "VIOLENCIA", "pattern": "violencia económica"},
        {"label": "VIOLENCIA", "pattern": "violencia patrimonial"},
    ]
    ruler.add_patterns(patrones)

    return nlp


# -----------------------------
# 3) Matcher adicional (fechas dd/mm/yyyy)
# -----------------------------
def build_matcher(nlp) -> Matcher:
    matcher = Matcher(nlp.vocab)
    # Ej: 05/01/2021, 02/02/2020
    date_pattern = [{"SHAPE": "dd/dd/dddd"}]
    matcher.add("FECHA_SLASH", [date_pattern])
    return matcher


def merge_entities(doc, extra_spans: List[Span]) -> List[Span]:
    """
    Une entidades del doc (doc.ents) con spans adicionales del matcher.
    Elimina duplicados por (start, end, label).
    """
    uniq: Dict[Tuple[int, int, str], Span] = {}

    for e in doc.ents:
        uniq[(e.start, e.end, e.label_)] = e

    for s in extra_spans:
        uniq[(s.start, s.end, s.label_)] = s

    # Ordenar por aparición
    return sorted(uniq.values(), key=lambda x: (x.start, x.end))


def spans_to_ents(doc, spans: List[Span]) -> None:
    """Sobrescribe doc.ents con los spans resultantes (para visualización/export)."""
    doc.ents = tuple(spans)


# -----------------------------
# 4) Ejecución: extracción + evidencias
# -----------------------------
def main() -> None:
    nlp = build_nlp()
    matcher = build_matcher(nlp)

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    resumen_txt = out_dir / "entidades_detectadas.txt"
    html_file = out_dir / "entidades_resaltadas.html"

    # Guardaremos docs procesados para exportar a HTML con entidades resaltadas
    docs_for_html = []

    with resumen_txt.open("w", encoding="utf-8") as f:
        f.write("Actividad 2 - NER jurídico (spaCy)\n")
        f.write("================================\n\n")

        for i, texto in enumerate(TEXTOS, start=1):
            doc = nlp(texto)

            # Matcher fechas dd/mm/yyyy
            matches = matcher(doc)
            extra_spans = []
            for _, start, end in matches:
                extra_spans.append(Span(doc, start, end, label="FECHA"))

            all_spans = merge_entities(doc, extra_spans)
            spans_to_ents(doc, all_spans)

            docs_for_html.append(doc)

            # Salida por consola + TXT
            f.write(f"Texto {i}:\n{texto}\n")
            if doc.ents:
                for ent in doc.ents:
                    f.write(f" - {ent.text} | {ent.label_}\n")
            else:
                f.write(" - (Sin entidades detectadas)\n")
            f.write("\n")

    # Export HTML para capturas (abre en navegador)
    options = {"ents": ["NORMA", "VIOLENCIA", "FECHA", "PER", "LOC", "ORG", "DATE"]}
    html = spacy.displacy.render(docs_for_html, style="ent", page=True, options=options)

    html_file.write_text(html, encoding="iso-8859-1")

    print("✅ Actividad 2 ejecutada correctamente.")
    print(f"📄 Evidencia TXT: {resumen_txt.resolve()}")
    print(f"🌐 Evidencia HTML (resaltado): {html_file.resolve()}")
    print("Sugerencia: abre el HTML en el navegador y toma capturas de entidades resaltadas.")


if __name__ == "__main__":
    main()
