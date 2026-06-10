# Informe Técnico — Marathon Tracker

**TP Final · Visión Artificial · Universidad Austral · 2026**
Clara Lopez & Santos Bogo

---

## 1. Descripción del sistema

Marathon Tracker es un sistema de visión artificial que registra automáticamente a los corredores al cruzar la línea de llegada de una maratón. A partir de un video de la llegada, el sistema identifica a cada corredor, lee el número de su dorsal y registra el timestamp exacto del cruce, produciendo una tabla de resultados sin intervención humana.

El pipeline completo es:

```
Video → Detección de personas → Detección de dorsal → OCR multi-frame → Registro de cruce
```

---

## 2. Detección y tracking de personas

Se utilizó **YOLOv8n** (variante nano de YOLOv8) con tracking integrado de Ultralytics. La elección se fundamenta en:

- **Tracking nativo**: YOLOv8 incluye un tracker que asigna IDs consistentes a cada persona a lo largo de los frames, sin necesidad de implementar un algoritmo de asociación externo.
- **Velocidad**: la variante nano corre en tiempo real en CPU, lo que permite procesar el video sin hardware especializado.
- **Filtrado por clase**: el modelo se configura para detectar únicamente la clase `persona` (`classes=[0]`), reduciendo falsos positivos.

El ROI (región de interés) definido interactivamente por el usuario limita el área de detección a la zona relevante del video, mejorando tanto la precisión como la performance.

---

## 3. Detección de dorsales

### 3.1 Comparación de modelos

Se evaluaron dos modelos para detectar los dorsales (bibs) dentro del bounding box de cada persona:

| Modelo | Origen | Observaciones |
|---|---|---|
| `bib-detector/2` | Roboflow (hosted, inference API) | Modelo genérico, detectaba con menor precisión en condiciones de movimiento y oclusión parcial |
| `bib_yolov8-1.pt` | Fine-tuning local con dataset de Roboflow | Mejor precisión en dorsales parcialmente tapados, en movimiento y con distintas iluminaciones |

El modelo `bib_yolov8-1.pt` resultó claramente superior, en particular para bibs en perspectiva, con motion blur o parcialmente cubiertos por brazos u otros corredores.

### 3.2 Entrenamiento local con fine-tuning

El modelo `bib_yolov8-1.pt` fue entrenado localmente mediante fine-tuning sobre YOLOv8. El proceso fue el siguiente:

**1. Dataset**: se utilizó el dataset público *"Bib Detection Big Data"* disponible en Roboflow Universe, con aproximadamente 600 imágenes anotadas de dorsales en contextos de carrera. Las anotaciones incluyen bounding boxes alrededor de cada dorsal.

**2. Descarga del dataset**: Roboflow permite descargar los datasets en formato YOLOv8 directamente via API. Esto genera una carpeta con imágenes y archivos `.txt` de labels, más un archivo `data.yaml` que describe las clases y las rutas de train/validation/test. Este archivo es el que consume YOLOv8 para entrenar.

**3. Fine-tuning**: en lugar de entrenar desde cero, se partió de los pesos preentrenados de YOLOv8n (`yolov8n.pt`), que ya tiene conocimiento general de formas, bordes y texturas aprendido sobre ImageNet y COCO. El fine-tuning especializa esos pesos para la tarea específica de detectar dorsales:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")          # pesos base preentrenados
model.train(data="data.yaml", epochs=50)
```

**4. Resultado**: después de 50 épocas el modelo produce `runs/detect/train/weights/best.pt`, que es el `bib_yolov8-1.pt` utilizado en producción.

**¿Por qué el fine-tuning mejora los resultados?** Un modelo genérico no vio dorsales de maratón durante su entrenamiento. Al fine-tunear con 600 imágenes específicas del dominio, el modelo aprende las características visuales propias de los bibs: fondo blanco con números negros, forma rectangular, posición típica en el torso. Esto reduce tanto los falsos positivos (detectar como dorsal algo que no lo es) como los falsos negativos (no detectar un dorsal real).

---

## 4. Lectura del número de dorsal (OCR)

### 4.1 Comparación de estrategias

Se compararon tres motores de OCR sobre los crops de bibs detectados. Para hacer la comparación justa, todos reciben el mismo preprocesamiento: upscale 3x con interpolación bicúbica + conversión a escala de grises + CLAHE (ecualización adaptativa de histograma) para mejorar el contraste.

| Motor | Tipo | Ventajas | Desventajas observadas |
|---|---|---|---|
| **Tesseract** | Clásico (OCR tradicional) | Rápido, sin dependencias pesadas | Muy sensible a perspectiva, ruido y motion blur; requiere imagen limpia |
| **EasyOCR** | Deep learning (CRAFT + CRNN) | Tolera condiciones reales mejor que Tesseract; reconoce texto en ángulos y con fondo complejo | Ocasionalmente falla en números de un solo dígito |
| **PaddleOCR** | Deep learning (DB + SVTR) | Mayor precisión general; mejor en dígitos con fuentes variadas | Más pesado; requiere Python ≤ 3.12 |

**Conclusión**: PaddleOCR obtuvo la mayor tasa de lectura correcta. EasyOCR fue más robusto en casos difíciles donde PaddleOCR no detectaba texto (dorsales muy pequeños o muy inclinados). Tesseract quedó descartado por su fragilidad ante las condiciones reales del video.

### 4.2 Estrategia combinada: voting multi-frame

En lugar de elegir un único motor, se diseñó una estrategia que combina PaddleOCR y EasyOCR usando voting acumulado a lo largo de múltiples frames:

**¿Por qué voting multi-frame?** Un corredor aparece en decenas de frames antes de cruzar la línea. En cada frame la lectura puede variar por motion blur, oclusión parcial o cambios de iluminación. Acumular lecturas y votar por la más frecuente produce un resultado mucho más robusto que confiar en la lectura de un único frame.

**Lógica de ponderación por frame:**

```
Si Paddle y EasyOCR leen el mismo número  →  sumar 2 votos a ese número
Si leen números distintos                 →  sumar 1 voto a cada uno
Si solo uno lee algo                      →  sumar 1 voto al que leyó
```

La coincidencia entre motores es una señal fuerte de que la lectura es correcta, por eso se pondera con doble peso. Al momento del cruce de línea, se toma el número con mayor cantidad de votos acumulados, exigiendo un mínimo de 3 votos para evitar registrar lecturas con poca evidencia.

---

## 5. Detección de cruce de línea

La línea de llegada es definida por el usuario haciendo clic en dos puntos sobre el primer frame. La detección del cruce usa geometría analítica: se calcula si el pie del corredor (centro inferior del bounding box) superó la línea en el eje perpendicular a ella. Para evitar registros duplicados, cada ID de tracking se agrega a un conjunto `crossed_ids` la primera vez que cruza.

---

## 6. Resumen de decisiones tecnológicas

| Componente | Tecnología elegida | Alternativas consideradas |
|---|---|---|
| Detección de personas | YOLOv8n con tracking integrado | Detectores sin tracking + algoritmo externo (SORT, DeepSORT) |
| Detección de dorsales | YOLOv8n fine-tuneado (local) | Modelo genérico de Roboflow (inferior en precisión) |
| OCR | PaddleOCR + EasyOCR combinados | Tesseract (descartado), un solo motor (menor robustez) |
| Voting | Multi-frame con ponderación por coincidencia | Single-frame (muy sensible a ruido) |
