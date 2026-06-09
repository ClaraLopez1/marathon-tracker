# 🏃 Marathon Tracker - TP Final — Visión Artificial · Universidad Austral

#### Clara Lopez & Santos Bogo

Sistema de registro automático de corredores en maratón usando visión artificial.

## ¿Qué hace?
Dado un video de la línea de llegada:
1. El usuario define interactivamente el ROI y la línea de llegada
2. Detecta y trackea personas con YOLOv8
3. Detecta los dorsales de cada corredor con un modelo YOLO fine-tuneado
4. Lee el número del dorsal combinando PaddleOCR y EasyOCR con voting multi-frame
5. Registra posición, número de dorsal y timestamp exacto de cada cruce

## Salida
Por cada corredor que cruza la línea se imprime:
```
Posición 1 | ID 3 | Bib 1042 | Tiempo: 00:14
```

## Tecnologías
- **Python 3.12**
- **OpenCV** — video I/O, visualización, preprocesamiento de imágenes
- **YOLOv8** (ultralytics) — detección y tracking de personas
- **YOLOv8 fine-tuneado** (`bib_yolov8-1.pt`) — detección de dorsales
- **PaddleOCR** + **EasyOCR** — lectura de números con voting multi-frame

## Estructura
```
src/
├── main.py            — loop principal de video, lógica de cruce
├── person_detector.py — detección y tracking de personas (YOLOv8)
├── bib_reader.py      — detección de dorsal + OCR + voting multi-frame
├── roi_selector.py    — selección interactiva del ROI
└── finish_line.py     — definición interactiva de la línea de llegada
models/
├── yolov8n.pt         — modelo base de personas
└── bib_yolov8-1.pt    — modelo de detección de dorsales
```


## Setup

### Activar venv con Python 3.12
```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

### Instalar dependencias
```sh
pip install -r requirements.txt
```

```sh
brew install ffmpeg
```

### Ejecutar
```sh
python src/main.py <nombre_video>
```

# Ejemplo:
```sh
python src/main.py ellport.mp4
```

## Descargar videos

### Maratón en Ellport Community
```sh
yt-dlp -f "bestvideo" -o "videos/ellport.mp4" "https://www.youtube.com/watch?v=rJJ8hu-TDBU"
```

#### Recortar video:
```sh
ffmpeg -ss 00:08:04 -i "videos/ellport.mp4" -c copy "videos/ellport_cut.mp4"
mv videos/ellport_cut.mp4 videos/ellport.mp4
```

### Maratón "Born to Run"
```sh
yt-dlp -f "bestvideo" -o "videos/borntorun.mp4" "https://www.youtube.com/watch?v=MWNRColAEao"
```

#### Recortar video:
```sh
ffmpeg -ss 00:05:00 -i "videos/borntorun.mp4" -c copy "videos/borntorun_cut.mp4"
mv videos/borntorun_cut.mp4 videos/borntorun.mp4
```