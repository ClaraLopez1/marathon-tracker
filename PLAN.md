# Plan: Sistema de registro automático de llegada de corredores

## Context

El proyecto `marathon-tracker` hoy detecta personas con YOLOv8, permite definir un ROI
y una línea de llegada de forma interactiva, y registra el cruce de cada corredor con
timestamp y posición (`src/main.py`, `src/person_detector.py`, `src/roi_selector.py`,
`src/finish_line.py`). Lo que **falta** para completar el TP es la parte que da sentido
al registro: localizar el **dorsal** de cada corredor, rectificarlo con **homografía**,
**leer el número**, asociarlo al ID trackeado, y producir **estadísticas** de llegada.

El repo hermano `artificial-vision/` (otra carpeta, otro git) contiene implementaciones
de la cursada que sirven de base directa y deben **portarse** (no se pueden importar entre
repos): homografía, clasificación ML con Hu moments, y componentes conectados/contornos.

Objetivo: un único script que **funcione end-to-end**. Sin CSV ni informe — overlay en
pantalla durante el video y un **resumen de estadísticas impreso en consola** al finalizar.

## Decisiones de diseño (tomadas)

- **Detección de dorsal:** YOLOv8 **local** entrenado con un dataset público de Roboflow
  (bib/race-number detection). Offline en runtime, coincide con la consigna.
- **Lectura del número:** **Tesseract OCR** (`pytesseract`) con whitelist `0123456789`,
  precedido de segmentación por **componentes conectados/contornos** para aislar y limpiar
  los dígitos. Cubre ese concepto de la cursada manteniendo robustez.
- **Corrección de perspectiva:** homografía de 4 puntos, portando `tp4/homography.py`.
- **Asignación dorsal→ID:** **voto por mayoría** de las lecturas OCR a lo largo de los
  frames en que el corredor es visible, para robustez ante lecturas ruidosas.
- **Salida:** overlay en vivo + resumen por consola. Sin archivos.

## Conceptos de la cursada cubiertos

- Segmentación y detección → YOLO persona + YOLO dorsal + thresholding.
- Corrección de perspectiva → homografía sobre la región del dorsal.
- Componentes conectados y contornos → segmentación/limpieza de dígitos pre-OCR.
- Clasificación con ML → el modelo de detección de dorsal (entrenado).

## Archivos a crear / modificar (en `marathon-tracker/`)

Nuevos en `src/`:
- **`bib_detector.py`** — carga `weights/bib_yolov8.pt`; dado el frame y el bbox de una
  persona, corre YOLO de dorsal recortado a esa región y devuelve el bbox del dorsal
  (y, si el dataset lo permite, polígono/4 esquinas para la homografía; si no, se usa el
  bbox rectangular como fuente de los 4 puntos).
- **`perspective.py`** — portar `order_points`/`four_point_transform` de
  `artificial-vision/practica_homografia/homografia.py` y
  `compute_homography_from_points`/`sort_corners` de `artificial-vision/tp4/homography.py`.
  Función `rectify_bib(frame, corners, size)` → imagen frontal del dorsal.
- **`digit_reader.py`** — pipeline sobre la región rectificada:
  1. grises + `adaptive_threshold` (portado de `tp2/preprocessing.py`),
  2. `connectedComponentsWithStats` filtrando por w/h/área (patrón de
     `artificial-vision/practica_comp_conectados/utils.py`) para aislar/limpiar dígitos,
  3. `pytesseract.image_to_string(..., config="--psm 7 -c tessedit_char_whitelist=0123456789")`,
  4. devuelve string numérico o `None` si inválido.
- **`tracker.py`** — estado por `track_id`: acumula votos de lecturas OCR
  (`Counter`), marca cruce de línea, asigna número final (voto mayoritario) y posición.
  Mueve acá `crosses_line()` que hoy vive en `main.py`.
- **`statistics.py`** — al terminar el video: total de corredores, distribución de
  tiempos (bins de 1 min) y cantidad de corredores por franja, impreso en consola.
- **`visualization.py`** — dibujar ROI, línea, bbox persona, bbox dorsal y número leído.

Entrenamiento (offline, una sola vez):
- **`train_bib_model.py`** (raíz del repo) — usa `roboflow` SDK para bajar un dataset
  público de dorsales y `ultralytics` para entrenar → `weights/bib_yolov8.pt`.
  Documentar en comentarios que requiere API key de Roboflow solo para esta etapa.

Modificar:
- **`src/main.py`** — orquestar el pipeline: por cada frame, detectar personas
  (`person_detector`), por cada persona detectar dorsal (`bib_detector`), rectificar
  (`perspective`), leer número (`digit_reader`), actualizar `tracker`; al cruzar la línea
  finalizar registro; al final invocar `statistics`. Quitar `crosses_line()` (va a tracker).
- **`requirements.txt`** (hoy vacío) — `opencv-python`, `numpy`, `ultralytics`,
  `pytesseract`, `roboflow` (solo entrenamiento). Nota: Tesseract requiere el binario del
  sistema (`brew install tesseract`).
- **`README.md` / `CLAUDE.md`** — actualizar pipeline y comandos.

## Reutilización concreta (portar, no importar)

- `artificial-vision/tp4/homography.py` → `src/perspective.py`
- `artificial-vision/practica_homografia/homografia.py` (`four_point_transform`) → idem
- `artificial-vision/tp2/preprocessing.py` (`adaptive_threshold`) → `src/digit_reader.py`
- `artificial-vision/practica_comp_conectados/utils.py` (filtros w/h/área) → `src/digit_reader.py`

## Verificación end-to-end

1. `brew install tesseract` y `pip install -r requirements.txt`.
2. Entrenar/obtener pesos: `python train_bib_model.py` (genera `weights/bib_yolov8.pt`).
   Si no hay key/dataset a mano, permitir un path alternativo a pesos existentes vía
   variable de entorno o constante.
3. Colocar `data/video.mp4` y correr `python src/main.py`.
4. Definir ROI y línea (flujo interactivo ya existente).
5. Validar: cada corredor muestra su número de dorsal sobre el bbox; al cruzar la línea
   se imprime `Posición N | Dorsal XXXX | tiempo MM:SS`; al terminar el video se imprime
   el resumen estadístico (total, distribución por franja).
6. Caso borde: corredor sin dorsal legible → se registra cruce con dorsal "??" sin romper
   el loop; voto mayoritario evita asignar un número por una sola lectura ruidosa.
