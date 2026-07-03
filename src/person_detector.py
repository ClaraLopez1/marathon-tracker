import cv2
from pathlib import Path
from ultralytics import YOLO

_MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n.pt"
model = YOLO(str(_MODEL_PATH))


def detect_persons(frame, roi=None):
    """
    Corre YOLO con tracking sobre el frame.
    Devuelve lista de (track_id, x1, y1, x2, y2).
    El ROI se usa solo para decidir si un corredor es válido la primera vez.
    Una vez trackeado, se mantiene su ID aunque salga del ROI.
    """
    results = model.track(frame, classes=[0], persist=True, verbose=False)
    detections = []

    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue

        for i in range(len(r.boxes)):
            box = r.boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            tid = int(r.boxes.id[i])

            if conf < 0.4:
                continue

            if roi is not None and not is_in_roi((x1, y1, x2, y2), roi):
                continue

            detections.append((tid, x1, y1, x2, y2))

    return detections


def is_in_roi(box, roi):
    """Devuelve True si el centro del box está dentro del polígono ROI."""
    import numpy as np
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    poly = np.array(roi, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), measureDist=False) >= 0