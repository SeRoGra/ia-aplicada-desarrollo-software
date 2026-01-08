# Actividad 2 – Diccionario de Términos y Clasificación Básica

## Descripción
Esta actividad corresponde al **Caso Práctico de la Unidad 1** y tiene como objetivo construir un **diccionario de términos jurídicos** y desarrollar un **mecanismo de clasificación básica** basado en reglas.

La solución permite identificar de forma preliminar la categoría jurídica más probable de un texto, utilizando coincidencias de palabras clave previamente definidas.

---

## Objetivo
- Definir categorías jurídicas relevantes.
- Asociar palabras clave representativas a cada categoría.
- Implementar una función de clasificación simple basada en conteo de coincidencias.
- Simular una orientación inicial para un consultorio jurídico.

---

## Estructura del proyecto

```
Actividad-2/
├── src/
│   ├── __init__.py
│   └── clasificacion.py
└── data/
    └── diccionario_legal.json
```

---

## Requisitos previos

- Python 3.11 o superior
- Entorno virtual activo (recomendado)
- pyenv (gestor de versiones)

---

## Archivo de entrada

### Diccionario de términos jurídicos

El archivo de entrada corresponde a un diccionario en formato JSON que contiene las categorías jurídicas y sus respectivas palabras clave.

**Ubicación:**
```
Actividad-2/data/diccionario_legal.json
```

**Estructura esperada:**
```json
{
  "familia": ["custodia", "alimentos", "divorcio"],
  "laboral": ["contrato", "salario", "despido"]
}
```

Cada categoría contiene entre **8 y 15 términos**, de acuerdo con lo solicitado en la actividad.

---

## Ejecución del script

1. Ubicarse en la carpeta raíz de la actividad:

```bash
cd Actividad-2
```

2. Ejecutar el módulo de clasificación:

```bash
python -m src.clasificacion
```

3. Ingresar un texto jurídico cuando el sistema lo solicite.

---

## Resultado esperado

El programa mostrará en consola:

- La **categoría jurídica estimada**
- El **detalle de puntuaciones** por cada categoría

Ejemplo de salida:

```text
Categoría estimada: familia
Detalle de puntuaciones: {'familia': 3, 'laboral': 0, 'penal': 0}
```

---

## Consideraciones

- El método de clasificación es **basado en reglas**, no en modelos entrenados.
- La solución tiene un propósito **académico y demostrativo**.
- No reemplaza el análisis jurídico profesional.
- Sirve como apoyo preliminar dentro del contexto del caso práctico.

---

## Autor

**Sebastian Rodríguez Granja**  
Caso Práctico 1 – IA Aplicada al Desarrollo de Software  
Universidad Asturias
