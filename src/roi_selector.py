import cv2
import numpy as np

_poly_points = []
_cursor = (0, 0)


def _on_mouse(event, x, y, flags, param):
    global _poly_points, _cursor
    _cursor = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _poly_points.append((x, y))


def select_roi(frame):
    """El usuario hace click para definir los vértices del ROI poligonal.
    Enter confirma (mínimo 3 puntos). Backspace deshace el último punto."""
    global _poly_points, _cursor
    _poly_points = []
    _cursor = (0, 0)

    clone = frame.copy()
    cv2.namedWindow("Seleccionar ROI")
    cv2.setMouseCallback("Seleccionar ROI", _on_mouse)

    print("Hacé click para agregar vértices del ROI. Backspace para deshacer. Enter para confirmar (mínimo 3 puntos).")

    while True:
        display = clone.copy()

        if len(_poly_points) >= 2:
            pts = np.array(_poly_points, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=False, color=(255, 165, 0), thickness=2)

        # Línea desde el último punto hasta el cursor
        if len(_poly_points) >= 1:
            cv2.line(display, _poly_points[-1], _cursor, (255, 165, 0), 1)
            # Preview del cierre del polígono
            if len(_poly_points) >= 3:
                cv2.line(display, _poly_points[0], _cursor, (255, 165, 0), 1)

        # Dibujar vértices
        for pt in _poly_points:
            cv2.circle(display, pt, 5, (0, 200, 255), -1)

        n = len(_poly_points)
        msg = f"Vértices: {n}  |  Enter para confirmar" if n >= 3 else f"Vértices: {n}  |  Necesitás al menos 3"
        cv2.putText(display, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Seleccionar ROI", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(_poly_points) >= 3:  # Enter
            break
        if key in (8, 127) and _poly_points:      # Backspace
            _poly_points.pop()
        if key == 27:                              # Esc
            _poly_points = []
            break

    cv2.destroyWindow("Seleccionar ROI")

    return _poly_points if len(_poly_points) >= 3 else None
