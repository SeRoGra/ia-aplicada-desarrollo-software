# Actividad 1 – Preprocesamiento de Corpus Jurídico (NLP)

## Descripción
Esta actividad corresponde al **Caso Práctico de la Unidad 1** y tiene como objetivo realizar el **preprocesamiento de un corpus jurídico** utilizando técnicas básicas de Procesamiento de Lenguaje Natural (NLP).

El proceso incluye:
- Normalización de texto
- Limpieza de caracteres
- Tokenización
- Eliminación de stopwords
- Lematización
- Generación de un corpus limpio en formato CSV

El corpus utilizado es **simulado**, cumpliendo con los lineamientos académicos del ejercicio.

---

## Estructura del proyecto

```
Actividad-1/
├── src/
│   └── preprocesamiento.py
└── data/
│   └── corpus_original.json
│   └── corpus_limpio.csv      # Output (generado automáticamente)
```

---

## Requisitos previos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- pyenv (gestor de versiones)

---

## Instalación de dependencias locales

Ejecutar los siguientes comandos **una sola vez**:

```bash
python -m pip install -U pip
python -m pip install spacy
python -m spacy download es_core_news_sm
```

> Si tu sistema usa `python3` en lugar de `python`, reemplaza el comando según corresponda.

---

## Archivo de entrada (Input)

El archivo de entrada debe llamarse:

```
corpus_original.json
```

Ubicación obligatoria:

```
Actividad-1/data/corpus_original.json
```

Formato esperado:

```json
[
  {
    "id": 1,
    "category_hint": "familia",
    "text": "Texto jurídico simulado..."
  }
]
```

El archivo debe contener **al menos 50 fragmentos de texto jurídico**.

---

## Ejecución del script

1. Ubícate en la carpeta `Actividad-1`:

```bash
cd Actividad-1
```

2. Ejecuta el script de preprocesamiento:

```bash
python src/preprocesamiento.py
```

---

## Resultado esperado

Si la ejecución es correcta, se mostrará el siguiente mensaje en consola:

```text
OK -> generado: Actividad-1/data/corpus_limpio.json
```

Y se generará automáticamente el archivo:

```
data/corpus_limpio.json
```

Cada elemento del archivo de salida contiene la siguiente estructura:

```json
{
  "id": 1,
  "category_hint": "familia",
  "text_original": "Texto jurídico original...",
  "text_clean": "texto juridico limpio",
  "text_lemmatized": "texto juridico lematizar"
}
```

## Autor
Sebastian Rodríguez Granja.
Caso Practico 1 - IA Aplicada al Desarrollo de Software
Universidad Asturias