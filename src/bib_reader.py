from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from paddleocr import PaddleOCR
import easyocr

ROOT      = Path(__file__).parent.parent
BIB_MODEL = ROOT / "models" / "bib_yolov8-1.pt"

DIGITS       = "0123456789"
CONF_BIB     = 0.35
UPSCALE      = 3
MIN_VOTES    = 3   # votos mínimos para considerar una lectura válida


def _preprocess(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    big  = cv2.resize(crop, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def _digits_only(text: str) -> str:
    return "".join(c for c in text if c in DIGITS)


class BibReader:
    def __init__(self):
        print("Cargando detector de bibs...")
        self._detector = YOLO(str(BIB_MODEL))

        print("Cargando EasyOCR...")
        self._easy = easyocr.Reader(["en"], gpu=False, verbose=False)

        print("Cargando PaddleOCR...")
        self._paddle = PaddleOCR(use_angle_cls=False, lang="en",
                                  show_log=False, use_gpu=False)

    def detect_and_read(self, frame: np.ndarray, person_box: tuple) -> tuple[str | None, int]:
        """
        Detecta el bib dentro de person_box y corre ambos motores OCR.
        Retorna (numero, weight):
          - weight=2 si Paddle y EasyOCR coinciden
          - weight=1 si solo uno leyó o difieren (se retorna el de Paddle, o Easy si Paddle vacío)
          - (None, 0) si ninguno leyó nada
        Cuando difieren, el caller debe llamar add() dos veces con cada lectura y weight=1.
        """
        px1, py1, px2, py2 = person_box
        person_crop = frame[py1:py2, px1:px2]
        if person_crop.size == 0:
            return None, 0

        bib_results = self._detector.predict(person_crop, verbose=False, conf=CONF_BIB)
        bib_crop = None
        best_conf = 0.0
        for r in bib_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    bib_crop = person_crop[by1:by2, bx1:bx2]

        if bib_crop is None or bib_crop.size == 0:
            return None, 0

        gray = _preprocess(bib_crop)

        paddle_text = self._run_paddle(gray)
        easy_text   = self._run_easy(gray)

        if paddle_text and easy_text:
            if paddle_text == easy_text:
                return paddle_text, 2
            # difieren: señalamos con weight=-1 para que main maneje ambas lecturas
            return (paddle_text, easy_text), -1
        if paddle_text:
            return paddle_text, 1
        if easy_text:
            return easy_text, 1
        return None, 0

    def _run_paddle(self, gray: np.ndarray) -> str:
        result = self._paddle.ocr(gray, cls=False)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and line[1]]
        return _digits_only("".join(texts))

    def _run_easy(self, gray: np.ndarray) -> str:
        results = self._easy.readtext(gray, allowlist=DIGITS, detail=0, paragraph=False)
        return _digits_only("".join(results))


class BibVoter:
    def __init__(self):
        self._votes: dict[int, Counter] = {}

    def add(self, track_id: int, reading: str, weight: int = 1):
        if not reading:
            return
        if track_id not in self._votes:
            self._votes[track_id] = Counter()
        self._votes[track_id][reading] += weight

    def get_best(self, track_id: int) -> str | None:
        counter = self._votes.get(track_id)
        if not counter:
            return None
        best, count = counter.most_common(1)[0]
        return best if count >= MIN_VOTES else None

    def get_current(self, track_id: int) -> str | None:
        """Mejor candidato actual sin exigir mínimo de votos (para visualización en tiempo real)."""
        counter = self._votes.get(track_id)
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def reset(self, track_id: int):
        self._votes.pop(track_id, None)
