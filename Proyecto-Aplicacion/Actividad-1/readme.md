# Proyecto de Aplicación – Actividad 1
IA Aplicada al Desarrollo de Software – UniAsturias

Este directorio contiene la implementación de un **producto mínimo viable (MVP)** orientado a la clasificación automática de textos legales por temática, como apoyo al proceso de triage en el consultorio jurídico virtual JustIA.

La solución fue desarrollada como parte del proyecto de aplicación de la asignatura *IA Aplicada al Desarrollo de Software* y busca aplicar conceptos básicos de procesamiento de lenguaje natural (NLP) y aprendizaje supervisado en un contexto práctico y controlado.

* * *

## ¿Qué hace `notebook.py`?

El archivo `notebook.py` ejecuta un flujo completo de clasificación de texto que incluye:

1. Definición de un conjunto de datos sintético con textos jurídicos y su categoría temática.
2. Separación de los datos en conjuntos de entrenamiento y prueba.
3. Vectorización de los textos mediante TF-IDF.
4. Entrenamiento de un modelo de clasificación `LinearSVC`.
5. Evaluación del modelo usando métricas estándar.
6. Ejecución de una predicción de ejemplo, mostrando los términos más relevantes que influyen en la decisión del modelo.

Este enfoque permite no solo clasificar los textos, sino también aportar una primera capa de trazabilidad sobre el comportamiento del modelo.

* * *

## Requisitos

- Python 3.9 o superior
- pip
- Librerías:
  - pandas
  - scikit-learn

* * *

## Ejecución del proyecto

### 1. Crear y activar entorno virtual

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install pandas scikit-learn
```

### 3. Ejecutar el script

```bash
python Proyecto-Aplicacion/Actividad-1/notebook.py
```

También es posible ejecutar el archivo desde VSCode, asegurándose de que el intérprete seleccionado corresponda al entorno virtual `.venv`.

* * *

## Salida esperada

Al ejecutar el script, se mostrará en consola:

- Un reporte de clasificación con métricas de precisión, recall y F1-score.
- La matriz de confusión.
- Un ejemplo de predicción con:
  - Texto de entrada
  - Categoría temática sugerida
  - Términos más relevantes asociados a la decisión del modelo

* * *

## Notas técnicas

- El conjunto de datos utilizado es reducido y sintético, por lo que las métricas obtenidas no representan un sistema productivo.
- El objetivo principal del ejercicio es demostrar el flujo completo de una solución basada en IA aplicada al desarrollo de software.
- El modelo no reemplaza el criterio humano ni la asesoría legal profesional.

* * *

## Autor

Sebastian Rodríguez Granja  
Asignatura: IA Aplicada al Desarrollo de Software  
Corporación Universitaria de Asturias

* * *

## Nota final

Este proyecto tiene un propósito estrictamente académico y forma parte del proceso de aprendizaje de la asignatura.
