# 🏃 Marathon Tracker

Sistema de registro automático de corredores en maratón usando visión artificial.

## ¿Qué hace?
Dado un video de la línea de llegada, detecta los dorsales de los corredores,
lee el número de cada uno y registra el momento exacto de cruce.

## Salida
Tabla automática con número de dorsal, timestamp de cruce y posición de llegada.

## Tecnologías
- Python 3
- OpenCV
- Tesseract OCR
- Roboflow (modelo de detección de dorsales)

## Estructura
- `src/main.py` — loop principal de video
- `src/finish_line.py` — definición de línea con mouse
- `src/bib_detector.py` — detección del dorsal
- `src/ocr.py` — lectura del número
- `src/tracker.py` — lógica de cruce y tiempos
- `src/visualization.py` — anotaciones sobre el frame

## TP Final — Visión Artificial · Universidad Austral
Clara Lopez & Santos Bogo · 2026