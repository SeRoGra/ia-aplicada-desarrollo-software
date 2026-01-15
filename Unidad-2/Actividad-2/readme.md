# Actividad 2 – Extractor de Entidades Nombradas (NER) Jurídico con spaCy
**IA Aplicada al Desarrollo de Software – Unidad 2**  
**Proyecto: JustIA**

## Descripción
Este entregable implementa un **extractor de entidades nombradas (NER)** para texto jurídico en español,
utilizando **spaCy** y un modelo base (`es_core_news_md`).
Se complementa el modelo con **patrones basados en reglas** para detectar entidades del dominio jurídico:

- **NORMA**: referencias a normas jurídicas (ej. *Ley 100 de 1993*)
- **VIOLENCIA**: tipos de violencia (ej. *violencia intrafamiliar*, *violencia económica*, *violencia patrimonial*)
- **FECHA**: fechas en formato `dd/mm/yyyy` mediante `Matcher`

Además, el script genera evidencias para la entrega:
- Un archivo `.txt` con el registro de entidades detectadas.
- Un archivo `.html` con entidades resaltadas (visualización estilo displaCy) para capturas.

---

## Requisitos
- macOS / Linux / Windows
- Python **3.11 o 3.12** (recomendado 3.12)
- `pyenv` (opcional, recomendado si tu Python por defecto es 3.14+)

> Nota: En Python 3.14 actualmente pueden presentarse incompatibilidades con dependencias internas de spaCy.

---

## Estructura esperada
```text
Actividad-2/
├── extractor_entidades_nombradas.py
├── README.md
├── .python-version            # opcional (pyenv)
├── .venv/                     # entorno virtual
└── output/
    ├── entidades_detectadas.txt
    └── entidades_resaltadas.html
```

---

## 1) Seleccionar Python 3.12 con pyenv (recomendado)
Si tu `python3 --version` es 3.14+, usa `pyenv` para fijar Python 3.12 solo en esta carpeta:

```bash
cd "Unidad-2/Actividad-2"
pyenv local 3.12.12
python --version
```

Debes ver: `Python 3.12.12`

---

## 2) Crear y activar entorno virtual
Desde la carpeta `Actividad-2`:

```bash
deactivate 2>/dev/null
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

En Windows:
```bash
.venv\Scripts\activate
```

---

## 3) Instalar dependencias
Instalar spaCy:

```bash
pip install spacy
```

Descargar el modelo en español:

```bash
python -m spacy download es_core_news_md
```

---

## 4) Ejecutar el script
Con el entorno virtual activado:

```bash
python extractor_entidades_nombradas.py
```

---

## 5) Evidencias generadas
Al finalizar, se crea/actualiza la carpeta `output/` con:

- `output/entidades_detectadas.txt`  
  Registro estructurado de entidades detectadas por cada texto.

- `output/entidades_resaltadas.html`  
  Visualización con entidades resaltadas (abre en navegador para capturas).

Abrir el HTML en macOS/Linux:

```bash
open output/entidades_resaltadas.html
```

En Windows (PowerShell):

```powershell
start output\entidades_resaltadas.html
```

---

## Autor
Sebastián Rodríguez Granja  
IA Aplicada al Desarrollo de Software – Unidad 2  
Corporación Universitaria de Asturias
