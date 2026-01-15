# Actividad 3 – Prototipo de Preguntas y Respuestas Jurídicas (QA)

## Descripción general
Esta actividad implementa un **prototipo de sistema de Preguntas y Respuestas jurídicas** basado en un enfoque **RAG (Retrieval-Augmented Generation) ligero**.  
El sistema permite responder preguntas en lenguaje natural utilizando una **base de conocimiento jurídica local**, recuperando los fragmentos más relevantes y mostrando evidencia de soporte.

---

## Estructura del proyecto

```
Actividad-3/
│
├── prototipo_question_answer_juridico.py
├── knowledge_base/
│   ├── 01_acoso_laboral_guia.txt
│   ├── 02_violencia_intrafamiliar_resumen.txt
│   ├── 03_conciliacion_extrajudicial.txt
│   ├── 04_derechos_migrantes_orientacion.txt
│   ├── 05_derechos_victimas_conflicto.txt
│   └── 06_derecho_familia_alimentos.txt
│
├── output/
│   └── evidencia_actividad3_qa.txt
│
├── .venv/
└── readme.md
```

---

## Requisitos

- Python **3.10 o 3.11** (recomendado)
- Sistema operativo: macOS, Linux o Windows
- Conexión a internet (solo la primera vez, para descargar modelos)

---

## Creación del entorno virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## Instalación de dependencias

```bash
pip install -U torch transformers sentence-transformers scikit-learn numpy
```

---

## Ejecución del sistema

Desde la carpeta `Actividad-3`:

```bash
python prototipo_question_answer_juridico.py
```

---

## Funcionamiento del sistema

1. Carga los documentos jurídicos desde `knowledge_base/`
2. Divide los textos en fragmentos (chunks)
3. Genera embeddings semánticos
4. Recupera los fragmentos más relevantes según la pregunta
5. Ejecuta un modelo de **Question Answering en español**
6. Muestra:
   - Pregunta
   - Respuesta generada
   - Puntaje de confianza
   - Fuentes recuperadas (top-k)
7. Guarda la evidencia en `output/evidencia_actividad3_qa.txt`

---

## Ejemplo de salida

```
Pregunta: ¿Qué hago si termine mi relación con mi novia y nunca firmé capitulaciones?
Respuesta: conciliacion_extrajudicial
Score: 0.0001

Fuentes recuperadas:
- 03_conciliacion_extrajudicial.txt
- 05_derechos_victimas_conflicto.txt
- 02_violencia_intrafamiliar_resumen.txt
```

---

## Evidencia generada

El sistema genera automáticamente un archivo de evidencia:

```
output/evidencia_actividad3_qa.txt
```

Este archivo contiene:
- La pregunta realizada
- La respuesta del sistema
- El nivel de confianza
- Los fragmentos documentales utilizados

---

## Observaciones académicas

- El prototipo es **experimental** y de carácter educativo
- No reemplaza asesoría legal profesional
- Demuestra el uso de IA aplicada al desarrollo de software jurídico
- Implementa principios de **explicabilidad y trazabilidad**

---

## Autor
Sebastián Rodríguez Granja  
Unidad 2 – IA Aplicada al Desarrollo de Software  
Actividad 3
