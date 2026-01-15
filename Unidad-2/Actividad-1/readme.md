# Actividad 1 – Clasificación de Fragmentos Jurídicos con BERT
**IA Aplicada al Desarrollo de Software – Unidad 2**  
**Proyecto: JustIA**

## Descripción
Este proyecto implementa el ajuste fino (*fine-tuning*) de un modelo BERT en español
(`dccuchile/bert-base-spanish-wwm-cased`) para la clasificación automática de fragmentos
de texto jurídico en cuatro áreas del derecho:

- Penal  
- Civil  
- Laboral  
- Familia  

El objetivo es evaluar la viabilidad técnica del uso de modelos de Procesamiento de Lenguaje
Natural (NLP) como fase intermedia de automatización del sistema JustIA.

---

## Requisitos del sistema
- Sistema operativo: macOS, Linux o Windows
- Python 3.10 o superior (recomendado 3.10 – 3.12)
- Conexión a internet para descargar el modelo preentrenado

---

## Creación y activación del entorno virtual (recomendado)

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows:

```bash
.venv\Scripts\activate
```

Actualizar pip:

```bash
pip install --upgrade pip
```

---

## Instalación de dependencias

Instalar las librerías necesarias para la ejecución del proyecto:

```bash
pip install torch transformers datasets scikit-learn
```

---

## Estructura del proyecto

```text
Actividad-1/
├── clasificacion_fragmentos_juridicos.py
├── README.md
└── .venv/
```

---

## Ejecución del script

Con el entorno virtual activado, ejecutar:

```bash
python clasificacion_fragmentos_juridicos.py
```

---

## Resultados esperados

Durante la ejecución se mostrarán en consola:
- Distribución de clases
- Pérdida por época
- Métricas finales (Accuracy y F1 macro)
- Ejemplos de predicción automática

---

## Autor
Sebastián Rodríguez Granja  
IA Aplicada al Desarrollo de Software – Unidad 2  
Corporación Universitaria de Asturias
