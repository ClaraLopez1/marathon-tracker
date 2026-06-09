"""
Compara dos modelos de detección de bibs en el mismo video.
  Verde  → bib_yolov8-1.pt     (ultralytics, local)
  Rojo   → bib-detector/2      (roboflow inference, cacheado en /tmp/cache)

Correr con: python bib_detector_model_compare.py

Controles: 'q' salir | SPACE pausar/resumir
"""

from pathlib import Path
from ultralytics import YOLO
from inference import get_model
import cv2

# ─── Config ───────────────────────────────────────────────────────────────────
 
ROOT = Path(__file__).parent.parent
VIDEO_PATH = ROOT / "data" / "honolulu.mp4"
MODELS_DIR = ROOT / "models"

MODEL_A_PATH = MODELS_DIR / "bib_yolov8-1.pt"

ROBOFLOW_API_KEY = "N337MrbrRDEqdpWVm0Hc"   # solo para el handshake, no re-descarga

COLOR_A = (0, 220, 0)    # verde → bib_yolov8-1
COLOR_B = (0, 60, 255)   # rojo  → bib-detector

CONF_THRESHOLD = 0.3
LABEL_A = "yolov8-1"
LABEL_B = "bib-detector"

# ─── Draw helpers ─────────────────────────────────────────────────────────────

def draw_ultralytics(frame, results, color, label_prefix):
    """Dibuja detecciones de ultralytics YOLO."""
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = r.names.get(int(box.cls[0]), "?")
            _draw_box(frame, x1, y1, x2, y2, conf, cls_name, color, label_prefix)


def draw_roboflow(frame, predictions, color, label_prefix):
    """Dibuja detecciones del paquete inference (coordenadas centrales)."""
    for pred in predictions:
        if pred.confidence < CONF_THRESHOLD:
            continue
        x1 = int(pred.x - pred.width  / 2)
        y1 = int(pred.y - pred.height / 2)
        x2 = int(pred.x + pred.width  / 2)
        y2 = int(pred.y + pred.height / 2)
        _draw_box(frame, x1, y1, x2, y2, pred.confidence, pred.class_name,
                  color, label_prefix)


def _draw_box(frame, x1, y1, x2, y2, conf, cls_name, color, label_prefix):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label_prefix} {cls_name} {conf:.2f}"
    cv2.putText(frame, text, (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Cargando modelos...")

    model_a = YOLO(str(MODEL_A_PATH))
    print(f"  ✓ A: {MODEL_A_PATH.name}")

    # get_model usa el caché local en /tmp/cache — no re-descarga
    model_b = get_model(model_id="bib-detector/2", api_key=ROBOFLOW_API_KEY)
    print(f"  ✓ B: bib-detector/2 (desde caché)")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {VIDEO_PATH}")
        return

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    paused  = False
    frame   = None

    print(f"\nVideo: {VIDEO_PATH.name}  |  {fps:.0f} fps")
    print("Controles: 'q' salir | SPACE pausar/resumir\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            # Inferencia
            res_a = model_a.predict(frame, verbose=False)
            res_b = model_b.infer(frame)

            # Dibujar
            draw_ultralytics(frame, res_a, COLOR_A, LABEL_A)

            preds_b = res_b[0].predictions if res_b else []
            draw_roboflow(frame, preds_b, COLOR_B, LABEL_B)

            # Leyenda
            cv2.rectangle(frame, (8, 8), (240, 54), (20, 20, 20), -1)
            cv2.putText(frame, f"Verde: {LABEL_A}", (14, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_A, 1)
            cv2.putText(frame, f"Rojo:  {LABEL_B}", (14, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_B, 1)

        if frame is not None:
            cv2.imshow("Bib Detector Compare", frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("Pausado" if paused else "Resumido")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()