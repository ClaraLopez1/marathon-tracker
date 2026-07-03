import cv2
import sys
import numpy as np
from pathlib import Path
from finish_line import select_finish_line
from roi_selector import select_roi
from person_detector import detect_persons
from bib_reader import BibReader, BibVoter

VIDEOS_DIR = Path(__file__).parent.parent / "videos"


def get_video_path() -> Path:
    if len(sys.argv) < 2:
        print("Uso: python main.py <nombre_video>")
        print(f"Videos disponibles en {VIDEOS_DIR}:")
        for f in sorted(VIDEOS_DIR.iterdir()):
            print(f"  {f.name}")
        sys.exit(1)
    return VIDEOS_DIR / sys.argv[1]


def crosses_line(box, line):
    """Devuelve True si el pie del corredor cruzó la línea."""
    x1, y1, x2, y2 = box
    foot_x = (x1 + x2) // 2
    foot_y = y2

    (lx1, ly1), (lx2, ly2) = line
    if lx2 == lx1:
        return False
    t = (foot_x - lx1) / (lx2 - lx1)
    if not (0 <= t <= 1):
        return False
    line_y = ly1 + t * (ly2 - ly1)
    return foot_y >= line_y


def main():
    video_path = get_video_path()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video abierto. {width}x{height} | FPS: {fps:.1f}")

    ret, first_frame = cap.read()
    if not ret:
        sys.exit(1)

    roi = select_roi(first_frame)
    if roi is None:
        print("No se definió el ROI. Saliendo.")
        sys.exit(1)
    print(f"ROI definido: {roi}")

    line = select_finish_line(first_frame)
    if line is None:
        print("No se definió la línea. Saliendo.")
        sys.exit(1)
    print(f"Línea definida: {line[0]} → {line[1]}")

    bib_reader = BibReader()
    voter      = BibVoter()

    crossed_ids        = set()           # IDs que ya cruzaron la línea (permanente)
    crossed_timestamps = {}              # tid → timestamp del cruce
    frames_absent      = {}              # tid → frames consecutivos fuera del ROI

    GRACE_FRAMES = 20                    # frames de gracia antes de considerar que salió

    frame_count = 0
    position = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video.")
            break

        frame_count += 1
        detections = detect_persons(frame, roi)
        active_ids_curr = {tid for tid, *_ in detections}

        # Actualizar contador de ausencia para corredores que cruzaron y no se ven
        for tid in list(crossed_timestamps):
            if tid not in active_ids_curr:
                frames_absent[tid] = frames_absent.get(tid, 0) + 1
                if frames_absent[tid] >= GRACE_FRAMES:
                    timestamp = crossed_timestamps.pop(tid)
                    frames_absent.pop(tid, None)
                    bib_number = voter.get_best(tid) or "?"
                    voter.reset(tid)
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    print(f"Posición {position} | ID {tid} | Bib {bib_number} | Tiempo: {minutes:02d}:{seconds:02d}")
                    position += 1
            else:
                frames_absent.pop(tid, None)  # reapareció, resetear contador

        # Dibujar ROI
        pts = np.array(roi, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 165, 0), thickness=2)

        # Dibujar línea
        cv2.line(frame, line[0], line[1], (0, 255, 0), 2)

        for tid, x1, y1, x2, y2 in detections:
            box = (x1, y1, x2, y2)
            is_crossed = crosses_line(box, line)

            # OCR en cada frame mientras el corredor esté en el ROI
            result, weight = bib_reader.detect_and_read(frame, box)
            if weight == -1:
                paddle_text, easy_text = result
                voter.add(tid, paddle_text, 1)
                voter.add(tid, easy_text, 1)
            elif result:
                voter.add(tid, result, weight)

            # Guardar timestamp de cruce (solo la primera vez)
            if is_crossed and tid not in crossed_ids:
                crossed_ids.add(tid)
                crossed_timestamps[tid] = frame_count / fps
                frames_absent.pop(tid, None)

            # Color según estado
            color = (0, 0, 255) if tid in crossed_ids else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid}"
            current_bib = voter.get_current(tid)
            if current_bib:
                label += f" #{current_bib}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Marathon Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()