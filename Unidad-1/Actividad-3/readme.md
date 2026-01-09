# Actividad 3 – Interfaz por Consola y Simulación de Consulta Jurídica

## Descripción
Esta actividad corresponde al **Caso Práctico de la Unidad 1** y tiene como objetivo desarrollar una **interfaz por consola** que permita simular la interacción de un usuario con un asistente jurídico denominado **JustIA**.

El sistema permite:
- Ingresar una consulta jurídica como texto libre.
- Cargar un documento de entrada en formato **.txt** o **.pdf**.
- Clasificar el contenido ingresado utilizando reglas simples.
- Mostrar una **respuesta simulada** con fines académicos.

---

## Objetivo
- Implementar un menú interactivo por consola.
- Permitir múltiples formas de entrada de información jurídica.
- Aplicar técnicas básicas de normalización de texto.
- Simular una orientación legal preliminar sin uso de modelos de IA entrenados.

---

## Estructura del proyecto

```
Actividad-3/
├── src/
│   ├── __init__.py
│   └── consola.py
└── data/
    └── diccionario_legal.json
```

---

## Requisitos previos

- Python 3.11 o superior
- Entorno virtual activo (recomendado)

Para la lectura de archivos PDF se requiere la siguiente dependencia adicional:

```bash
pip install PyPDF2
```

---

## Archivo de datos

### Diccionario de términos jurídicos

El sistema utiliza un diccionario de términos jurídicos almacenado en formato JSON.

**Ubicación:**
```
Actividad-3/data/diccionario_legal.json
```

Este archivo define las categorías jurídicas y las palabras clave asociadas a cada una.

---

## Ejecución del sistema

1. Ubicarse en la carpeta raíz de la actividad:

```bash
cd Actividad-3
```

2. Ejecutar el sistema por consola:

```bash
python -m src.consola
```

---

## Funcionalidades del menú

Al ejecutar el programa, se presenta un menú con las siguientes opciones:

1. Ingresar una pregunta legal como texto.
2. Cargar un documento de entrada (.txt o .pdf).
3. Clasificar el contenido cargado.
4. Visualizar una vista previa del contenido.
5. Salir del sistema.

---

## Resultado esperado

El sistema mostrará en consola:

- La fuente de entrada (texto o archivo).
- La **categoría jurídica estimada**.
- El detalle de puntuaciones por categoría.
- Una **respuesta simulada** de orientación legal.

Ejemplo:

```text
Categoría estimada: familia
Puntuaciones: {'familia': 3, 'laboral': 0, 'penal': 0}

El caso parece estar relacionado con derecho de familia...
```

---

## Consideraciones

- La clasificación se realiza mediante **reglas simples**, sin modelos de aprendizaje automático.
- La respuesta generada es únicamente **una simulación académica**.
- El sistema no reemplaza el análisis jurídico profesional.
- Esta actividad integra conceptos de procesamiento de texto vistos en actividades anteriores.

---

## Autor

**Sebastian Rodríguez Granja**  
Caso Práctico 1 – IA Aplicada al Desarrollo de Software  
Universidad Asturias
