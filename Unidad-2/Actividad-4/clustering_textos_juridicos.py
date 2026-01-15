# -*- coding: utf-8 -*-
"""
Actividad 4: Embeddings semánticos + Clustering de textos jurídicos

- Genera (simula) >= 100 fragmentos jurídicos clasificados por tema.
- Calcula embeddings con SentenceTransformers (SBERT multilingüe).
- Aplica KMeans (default) o DBSCAN (opcional).
- Visualiza los clusters con t-SNE (default) o UMAP (opcional).
- Exporta evidencias a output/ para capturas y entrega.

Ejecución:
    python clustering_textos_juridicos.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Clustering
CLUSTER_METHOD = "kmeans"   # "kmeans" o "dbscan"
KMEANS_K = 5                # número de clusters para KMeans
DBSCAN_EPS = 0.8            # ajusta si usas dbscan
DBSCAN_MIN_SAMPLES = 5

# Visualización
VIS_METHOD = "tsne"         # "tsne" o "umap" (umap es opcional)
TSNE_PERPLEXITY = 25
TSNE_ITER = 1500


@dataclass
class Sample:
    text: str
    label: str


def build_dataset(n_per_class: int = 25) -> List[Sample]:
    """
    Simula un dataset de textos jurídicos (>= 100) clasificados por tema.
    Total = n_per_class * num_temas
    """
    temas = ["penal", "civil", "laboral", "familia", "migratorio"]
    delitos = ["hurto", "lesiones personales", "estafa", "amenazas", "violencia intrafamiliar"]
    acciones = ["tutela", "demanda", "denuncia", "querella", "recurso"]
    entidades = ["Fiscalía", "Juzgado", "Comisaría de Familia", "Inspección del Trabajo", "Defensoría"]
    ciudades = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena"]
    normas = ["Ley 100 de 1993", "Ley 1098 de 2006", "Ley 1010 de 2006", "Ley 1257 de 2008", "Ley 1448 de 2011"]

    templates: Dict[str, List[str]] = {
        "penal": [
            "Se investiga el delito de {delito} y se ordena la práctica de pruebas en {ciudad}.",
            "La {entidad} imputó cargos por {delito} y solicitó medida de aseguramiento.",
            "En el proceso penal se analizan elementos materiales probatorios relacionados con {delito}.",
        ],
        "civil": [
            "Se presentó {accion} por incumplimiento contractual y reclamación de perjuicios.",
            "El litigio civil versa sobre responsabilidad extracontractual y reparación integral.",
            "El {entidad} resolvió controversia sobre propiedad, posesión y obligaciones.",
        ],
        "laboral": [
            "Se alega acoso laboral y se solicita reintegro por estabilidad reforzada.",
            "El trabajador interpuso {accion} por despido sin justa causa y salarios caídos.",
            "La {entidad} evaluó el conflicto por reconocimiento de prestaciones sociales.",
        ],
        "familia": [
            "Se tramita proceso de alimentos, custodia y regulación de visitas.",
            "La {entidad} ordenó medidas de protección por violencia intrafamiliar.",
            "Se analiza la filiación y el interés superior del menor conforme a {norma}.",
        ],
        "migratorio": [
            "La persona migrante solicita orientación sobre regularización y acceso a rutas de atención en {ciudad}.",
            "Se requiere asesoría para proteger derechos de población migrante y mecanismos de protección.",
            "El caso involucra verificación de permisos, atención humanitaria y enfoque diferencial.",
        ],
    }

    data: List[Sample] = []
    for tema in temas:
        for _ in range(n_per_class):
            t = random.choice(templates[tema]).format(
                delito=random.choice(delitos),
                accion=random.choice(acciones),
                entidad=random.choice(entidades),
                ciudad=random.choice(ciudades),
                norma=random.choice(normas),
            )
            # Variación extra para reducir “plantillas idénticas”
            t = f"{t} Referencia: {random.choice(normas)}."
            data.append(Sample(text=t, label=tema))

    random.shuffle(data)
    return data


def embed_texts(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return emb


def cluster_embeddings(emb: np.ndarray) -> np.ndarray:
    """
    Retorna labels de cluster para cada embedding.
    """
    if CLUSTER_METHOD == "kmeans":
        kmeans = KMeans(n_clusters=KMEANS_K, random_state=SEED, n_init="auto")
        return kmeans.fit_predict(emb)

    if CLUSTER_METHOD == "dbscan":
        # DBSCAN se beneficia de embeddings escalados
        emb_scaled = StandardScaler().fit_transform(emb)
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        return db.fit_predict(emb_scaled)

    raise ValueError("CLUSTER_METHOD debe ser 'kmeans' o 'dbscan'.")


def reduce_dimensionality(emb: np.ndarray) -> np.ndarray:
    """
    Convierte embeddings a 2D para graficar.
    """
    if VIS_METHOD == "tsne":
        tsne = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=min(TSNE_PERPLEXITY, max(5, (len(emb) // 5))),
            max_iter=TSNE_ITER,
            init="pca",
            learning_rate="auto",
        )
        return tsne.fit_transform(emb)

    if VIS_METHOD == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise ImportError("Para usar UMAP instala: pip install umap-learn") from e

        reducer = umap.UMAP(n_components=2, random_state=SEED)
        return reducer.fit_transform(emb)

    raise ValueError("VIS_METHOD debe ser 'tsne' o 'umap'.")


def save_outputs(samples: List[Sample], cluster_labels: np.ndarray, coords_2d: np.ndarray) -> Tuple[Path, Path, Path]:
    df = pd.DataFrame({
        "text": [s.text for s in samples],
        "tema_real": [s.label for s in samples],
        "cluster": cluster_labels,
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
    })

    csv_path = OUT_DIR / "clusters_dataset.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Resumen de clusters
    resumen_path = OUT_DIR / "clusters_resumen.txt"
    lines = []
    lines.append("Actividad 4 - Clustering semántico de textos jurídicos\n")
    lines.append("=====================================================\n\n")
    lines.append(f"Embeddings: {EMBED_MODEL}\n")
    lines.append(f"Clustering: {CLUSTER_METHOD}\n")
    if CLUSTER_METHOD == "kmeans":
        lines.append(f"KMeans K: {KMEANS_K}\n")
    else:
        lines.append(f"DBSCAN eps: {DBSCAN_EPS} | min_samples: {DBSCAN_MIN_SAMPLES}\n")
    lines.append(f"Visualización: {VIS_METHOD}\n\n")

    # Conteos por cluster y por tema
    lines.append("Distribución por cluster:\n")
    lines.append(df["cluster"].value_counts().sort_index().to_string())
    lines.append("\n\nCruce cluster vs tema_real:\n")
    crosstab = pd.crosstab(df["cluster"], df["tema_real"])
    lines.append(crosstab.to_string())
    lines.append("\n\nEjemplos por cluster (3):\n")

    for cl in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cl].head(3)
        lines.append(f"\n--- Cluster {cl} ---\n")
        for _, row in subset.iterrows():
            lines.append(f"* ({row['tema_real']}) {row['text']}\n")

    resumen_path.write_text("".join(lines), encoding="utf-8")

    # Plot
    plt.figure(figsize=(10, 6))
    # Matplotlib asigna colores automáticamente por cluster
    plt.scatter(df["x"], df["y"], c=df["cluster"], s=20)
    plt.title("Clustering semántico de textos jurídicos (2D)")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.grid(True, alpha=0.2)

    fig_name = "clusters_tsne.png" if VIS_METHOD == "tsne" else "clusters_umap.png"
    fig_path = OUT_DIR / fig_name
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    return csv_path, resumen_path, fig_path


def main() -> None:
    # 1) Dataset
    samples = build_dataset(n_per_class=25)  # 25 * 5 = 125 >= 100
    print(f"✅ Dataset generado: {len(samples)} textos")

    # 2) Embeddings
    texts = [s.text for s in samples]
    print("🔧 Generando embeddings...")
    emb = embed_texts(texts)
    print(f"✅ Embeddings generados: {emb.shape}")

    # 3) Clustering
    print("🔎 Aplicando clustering...")
    cluster_labels = cluster_embeddings(emb)
    print(f"✅ Clusters encontrados: {len(set(cluster_labels))} (incluye -1 si DBSCAN marca ruido)")

    # 4) Reducción a 2D
    print("📉 Reduciendo dimensionalidad para visualización...")
    coords_2d = reduce_dimensionality(emb)

    # 5) Salidas
    csv_path, resumen_path, fig_path = save_outputs(samples, cluster_labels, coords_2d)

    print("\n📌 Evidencias generadas:")
    print(f"- Dataset con clusters: {csv_path.resolve()}")
    print(f"- Resumen interpretativo: {resumen_path.resolve()}")
    print(f"- Gráfico 2D: {fig_path.resolve()}")


if __name__ == "__main__":
    main()