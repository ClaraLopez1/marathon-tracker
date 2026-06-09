"""
Compara tres estrategias de OCR sobre los bibs detectados por bib_yolov8-1.pt.

  Tesseract  → amarillo
  EasyOCR    → cian
  PaddleOCR  → magenta

Controles: 'q' salir | SPACE pausar/resumir | '+'/'-' velocidad

Instalación previa:
  pip install pytesseract easyocr paddleocr
  brew install tesseract          # macOS
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# OCR backends — importados con fallback para que el script corra aunque falte alguno
try:
    import pytesseract
    TESS_OK = True
except ImportError:
    TESS_OK = False
    print("[warn] pytesseract no instalado — Tesseract desactivado")

try:
    import easyocr 
    EASY_OK = True
except ImportError:
    EASY_OK = False
    print("[warn] easyocr no instalado — EasyOCR desactivado")

try:
    from paddleocr import PaddleOCR
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    PADDLE_OK = True
except Exception:
    PADDLE_OK = False
    print("[warn] paddleocr no disponible — PaddleOCR desactivado")

# ─── Config ───────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
VIDEOS_DIR = ROOT / "videos"
BIB_MODEL  = ROOT / "models" / "bib_yolov8-1.pt"

if len(sys.argv) < 2:
    print("Uso: python bib_ocr_compare.py <nombre_video>")
    print(f"Videos disponibles en {VIDEOS_DIR}:")
    for f in sorted(VIDEOS_DIR.iterdir()):
        print(f"  {f.name}")
    sys.exit(1)

VIDEO_PATH = VIDEOS_DIR / sys.argv[1]

CONF_BIB   = 0.35
UPSCALE    = 3       # factor para ampliar el crop antes del OCR
DIGITS     = "0123456789"

COLOR_TESS   = (0,   220, 220)   # amarillo
COLOR_EASY   = (220, 220,   0)   # cian
COLOR_PADDLE = (220,   0, 220)   # magenta
COLOR_BOX    = (255, 255, 255)   # blanco para el bbox del bib

# ─── Preprocesamiento del crop ────────────────────────────────────────────────

def preprocess(crop: np.ndarray) -> np.ndarray:
    """Upscale + gris + CLAHE. Mejora legibilidad para cualquier backend."""
    h, w = crop.shape[:2]
    big  = cv2.resize(crop, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)

# ─── OCR wrappers ─────────────────────────────────────────────────────────────

def ocr_tesseract(gray: np.ndarray) -> str:
    cfg = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
    raw = pytesseract.image_to_string(gray, config=cfg).strip()
    return "".join(c for c in raw if c in DIGITS) or "—"

def ocr_easyocr(gray: np.ndarray, reader) -> str:
    results = reader.readtext(gray, allowlist=DIGITS, detail=0, paragraph=False)
    merged  = "".join(results).strip()
    return merged if merged else "—"

def ocr_paddle(gray: np.ndarray, engine) -> str:
    # PaddleOCR 2.x necesita BGR (3 canales), no gris
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = engine.ocr(bgr, cls=False)

    # Manejo robusto: result puede ser [], [[]], [[None]] o None
    if not result or not result[0] or result[0] == [None]:
        return "—"

    texts = [line[1][0] for line in result[0] if line and line[1]]
    merged = "".join("".join(c for c in t if c in DIGITS) for t in texts)
    return merged if merged else "—"

# ─── Dibujo ───────────────────────────────────────────────────────────────────

def draw_bib(frame, x1, y1, x2, y2, labels: list[tuple[str, str, tuple]]):
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
    for i, (name, text, color) in enumerate(labels):
        line = f"{name}: {text}"
        ty   = y2 + 30 + i * 32
        cv2.putText(frame, line, (x1 + 2, ty + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, line, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

def draw_legend(frame):
    entries = [
        ("Tesseract",  COLOR_TESS),
        ("EasyOCR",    COLOR_EASY),
        ("PaddleOCR",  COLOR_PADDLE),
    ]
    cv2.rectangle(frame, (8, 8), (280, 20 + len(entries) * 36), (20, 20, 20), -1)
    for i, (name, color) in enumerate(entries):
        cv2.putText(frame, name, (14, 38 + i * 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Cargando detector de bibs...")
    detector = YOLO(str(BIB_MODEL))

    easy_reader = paddle_engine = None

    if EASY_OK:
        print("Cargando EasyOCR...")
        easy_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    if PADDLE_OK:
        print("Cargando PaddleOCR...")
        paddle_engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False, use_gpu=False)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {VIDEO_PATH}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    delay  = int(1000 / fps)
    paused = False
    frame  = None

    print(f"\nVideo: {VIDEO_PATH.name}  |  {fps:.0f} fps")
    print("Controles: 'q' salir | SPACE pausar/resumir | '+'/'-' velocidad\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            bib_results = detector.predict(frame, verbose=False, conf=CONF_BIB)

            for r in bib_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Guardar margen mínimo
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2 = min(x2, frame.shape[1])
                    y2 = min(y2, frame.shape[0])
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    gray = preprocess(crop)

                    labels = []
                    if TESS_OK:
                        labels.append(("Tess",   ocr_tesseract(gray),            COLOR_TESS))
                    if EASY_OK:
                        labels.append(("Easy",   ocr_easyocr(gray, easy_reader), COLOR_EASY))
                    if PADDLE_OK:
                        labels.append(("Paddle", ocr_paddle(gray, paddle_engine),COLOR_PADDLE))

                    draw_bib(frame, x1, y1, x2, y2, labels)

            draw_legend(frame)

        if frame is not None:
            cv2.imshow("Bib OCR Compare", frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("Pausado" if paused else "Resumido")
        elif key == ord("+"):
            delay = max(1, delay - 10)
        elif key == ord("-"):
            delay = min(500, delay + 10)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
