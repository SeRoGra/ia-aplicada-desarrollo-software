# -*- coding: utf-8 -*-
"""
Actividad 3: Prototipo QA Jurídico (RAG ligero)
- Base de conocimiento: archivos .txt en knowledge_base/
- Retrieval: SentenceTransformers + cosine similarity
- Reader QA: modelo en español (transformers pipeline)
- Entrega: respuesta + fuente (doc/chunk) + evidencia en output/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# -----------------------------
# Configuración
# -----------------------------
KB_DIR = Path("knowledge_base")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings (multilingüe, funciona bien para español)
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modelo QA en español (si falla, puedes cambiar por otro)
QA_MODEL = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"

TOP_K = 3
CHUNK_SIZE = 600      # caracteres por fragmento
CHUNK_OVERLAP = 80    # solapamiento entre fragmentos


# -----------------------------
# Estructuras
# -----------------------------
@dataclass
class Chunk:
    doc_name: str
    chunk_id: int
    text: str


# -----------------------------
# Utilidades
# -----------------------------
def ensure_sample_kb() -> None:
    """
    Si la carpeta knowledge_base/ está vacía, crea 6 textos base simulados
    (leyes/guías/resúmenes) para poder ejecutar y evidenciar.
    """
    KB_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(KB_DIR.glob("*.txt"))
    if existing:
        return

    samples = {
        "01_acoso_laboral_guia.txt": (
            "Guía básica sobre acoso laboral (Colombia).\n\n"
            "El acoso laboral comprende conductas persistentes y demostrables ejercidas sobre un trabajador, "
            "encaminadas a infundir miedo, intimidación, terror o angustia, causar perjuicio laboral, generar "
            "desmotivación o inducir la renuncia.\n\n"
            "Medidas recomendadas: documentar hechos (fechas, testigos), reportar al área de talento humano, "
            "usar comités de convivencia, y acudir a inspección del trabajo si persiste.\n"
        ),
        "02_violencia_intrafamiliar_resumen.txt": (
            "Resumen: violencia intrafamiliar.\n\n"
            "La violencia intrafamiliar puede incluir violencia física, psicológica, económica o patrimonial "
            "en el contexto familiar. Ante riesgo, se pueden solicitar medidas de protección.\n\n"
            "En contextos de atención a población vulnerable, es clave evitar revictimización y garantizar "
            "enfoque diferencial.\n"
        ),
        "03_conciliacion_extrajudicial.txt": (
            "Conciliación extrajudicial.\n\n"
            "En algunos asuntos, la conciliación puede ser un requisito de procedibilidad. "
            "Se recomienda verificar si el caso admite conciliación y ante qué autoridad.\n\n"
            "La conciliación busca soluciones acordadas y reduce congestión judicial.\n"
        ),
        "04_derechos_migrantes_orientacion.txt": (
            "Orientación general para personas migrantes.\n\n"
            "Las personas migrantes pueden acceder a rutas de atención y orientación jurídica, "
            "en especial cuando enfrentan vulneraciones de derechos. Es importante identificar "
            "la entidad competente y el mecanismo idóneo (tutela, denuncia, queja, etc.).\n"
        ),
        "05_derechos_victimas_conflicto.txt": (
            "Victimas del conflicto armado: orientación.\n\n"
            "Las víctimas pueden tener derecho a medidas de asistencia, atención, reparación y garantías "
            "de no repetición. Se recomienda orientar sobre rutas institucionales y acompañamiento.\n"
        ),
        "06_derecho_familia_alimentos.txt": (
            "Derecho de familia: alimentos.\n\n"
            "Las obligaciones alimentarias buscan garantizar el sustento de menores o personas dependientes. "
            "Puede existir acuerdo o fijación por autoridad competente. Se deben considerar capacidad económica "
            "y necesidades.\n"
        ),
    }

    for name, text in samples.items():
        (KB_DIR / name).write_text(text, encoding="utf-8")

    print(f"✅ Base de conocimiento creada en: {KB_DIR.resolve()}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())  # normaliza espacios
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n, d), b: (m, d)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)


def build_index(embedder: SentenceTransformer) -> Tuple[List[Chunk], np.ndarray]:
    chunks: List[Chunk] = []
    for doc_path in sorted(KB_DIR.glob("*.txt")):
        doc_text = doc_path.read_text(encoding="utf-8", errors="replace")
        parts = chunk_text(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, part in enumerate(parts, start=1):
            chunks.append(Chunk(doc_name=doc_path.name, chunk_id=i, text=part))

    if not chunks:
        raise RuntimeError("La base de conocimiento está vacía. Agrega archivos .txt en knowledge_base/.")

    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
    return chunks, embeddings


def retrieve(question: str, chunks: List[Chunk], chunk_emb: np.ndarray, embedder: SentenceTransformer, top_k: int) -> List[Tuple[Chunk, float]]:
    q_emb = embedder.encode([question], convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_sim_matrix(chunk_emb, q_emb).reshape(-1)  # (n,)
    top_idx = np.argsort(-sims)[:top_k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def answer_question(question: str, retrieved: List[Tuple[Chunk, float]], qa_pipe) -> dict:
    # Construye contexto uniendo los top chunks
    context = "\n\n".join([f"[{c.doc_name} :: chunk {c.chunk_id}] {c.text}" for c, _ in retrieved])

    result = qa_pipe(question=question, context=context)
    # result: {'score':..., 'start':..., 'end':..., 'answer':...}
    return {"answer": result.get("answer", ""), "score": float(result.get("score", 0.0)), "context": context}


def save_evidence(question: str, retrieved: List[Tuple[Chunk, float]], qa_result: dict) -> Path:
    out_path = OUT_DIR / "evidencia_actividad3_qa.txt"
    lines = []
    lines.append("Actividad 3 - Prototipo QA Jurídico (RAG ligero)\n")
    lines.append("================================================\n\n")
    lines.append(f"Pregunta del usuario:\n{question}\n\n")
    lines.append(f"Respuesta:\n{qa_result['answer']}\n")
    lines.append(f"Confianza (score): {qa_result['score']:.4f}\n\n")
    lines.append("Fuentes recuperadas (top-k):\n")
    for chunk, sim in retrieved:
        preview = (chunk.text[:200] + "...") if len(chunk.text) > 200 else chunk.text
        lines.append(f"- {chunk.doc_name} | chunk {chunk.chunk_id} | similitud={sim:.4f}\n  {preview}\n")
    lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_sample_kb()

    print("🔧 Cargando modelos...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # QA pipeline (CPU). Si tienes GPU, se puede configurar device=0.
    qa_pipe = pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL)

    print("📚 Indexando base de conocimiento...")
    chunks, chunk_emb = build_index(embedder)
    print(f"✅ Documentos: {len(list(KB_DIR.glob('*.txt')))} | Fragmentos indexados: {len(chunks)}")

    # Pregunta interactiva
    question = input("\nEscribe tu pregunta jurídica: ").strip()
    if not question:
        print("⚠️ Pregunta vacía. Finalizando.")
        return

    retrieved = retrieve(question, chunks, chunk_emb, embedder, TOP_K)
    qa_result = answer_question(question, retrieved, qa_pipe)

    print("\n==============================")
    print("✅ Respuesta del sistema")
    print("==============================")
    print(f"Pregunta: {question}")
    print(f"Respuesta: {qa_result['answer']}")
    print(f"Score: {qa_result['score']:.4f}\n")

    print("📌 Fuentes (top-k):")
    for c, sim in retrieved:
        print(f"- {c.doc_name} | chunk {c.chunk_id} | similitud={sim:.4f}")

    evidence_path = save_evidence(question, retrieved, qa_result)
    print(f"\n📄 Evidencia guardada en: {evidence_path.resolve()}")


if __name__ == "__main__":
    # Evita que transformers meta logs excesivos
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()