# Actividad 4 – Clustering semántico de textos jurídicos

## Descripción general
En esta actividad se implementa un proceso de **clustering no supervisado** aplicado a textos jurídicos, con el objetivo de identificar agrupamientos semánticos basados en similitud de contenido.
El enfoque combina **embeddings semánticos**, **algoritmos de clustering** y **reducción de dimensionalidad** para su análisis y visualización.

---

## Estructura del proyecto

```
Actividad-4/
│
├── clustering_textos_juridicos.py
├── output/
│   ├── clusters_dataset.csv
│   ├── clusters_resumen.txt
│   └── clusters_tsne.png
├── .venv/
└── readme.md
```

---

## Requisitos del entorno

- Python **3.11 o 3.12** (recomendado)
- Entorno virtual (`venv`)

### Librerías necesarias

```bash
pip install sentence-transformers scikit-learn matplotlib pandas tqdm
```

---

## Ejecución del código

1. Activar el entorno virtual:

```bash
source .venv/bin/activate
```

2. Ejecutar el script principal:

```bash
python clustering_textos_juridicos.py
```

---

## Proceso implementado

1. Generación de dataset simulado de textos jurídicos.
2. Cálculo de embeddings semánticos usando:
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
3. Clustering no supervisado mediante **K-Means**.
4. Reducción de dimensionalidad con **t-SNE** para visualización.
5. Generación de evidencias:
   - Dataset etiquetado por cluster
   - Resumen interpretativo
   - Gráfico bidimensional de clusters

---

## Archivos de salida

- **clusters_dataset.csv**  
  Dataset con textos, tema real, cluster asignado y coordenadas 2D.

- **clusters_resumen.txt**  
  Resumen del modelo utilizado, distribución de clusters y cruce con temáticas reales.

- **clusters_tsne.png**  
  Visualización bidimensional del clustering semántico.

---

## Observaciones técnicas

- El clustering es no supervisado, por lo que los clusters no corresponden exactamente a las categorías reales, pero muestran una alta coherencia semántica.
- La visualización mediante t-SNE permite identificar separaciones claras entre grupos temáticos.

---

## Autor
Sebastián Rodríguez Granja  
Unidad 2 – IA Aplicada al Desarrollo de Software  
Actividad 4