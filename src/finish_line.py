import cv2

line_points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
        line_points.append((x, y))
        print(f"Punto {len(line_points)}/2: ({x}, {y})")

def select_finish_line(frame):
    """Muestra el primer frame y permite al usuario clickear 2 puntos."""
    global line_points
    line_points = []

    clone = frame.copy()
    cv2.namedWindow("Seleccionar linea de llegada")
    cv2.setMouseCallback("Seleccionar linea de llegada", on_mouse)

    print("Clickeá 2 puntos para definir la línea de llegada. Enter para confirmar.")

    while True:
        display = clone.copy()

        for pt in line_points:
            cv2.circle(display, pt, 6, (0, 255, 0), -1)

        if len(line_points) == 2:
            cv2.line(display, line_points[0], line_points[1], (0, 255, 0), 2)

        cv2.imshow("Seleccionar linea de llegada", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(line_points) == 2:  # Enter
            break
        if key == 27:  # ESC
            break

    cv2.destroyWindow("Seleccionar linea de llegada")
    return tuple(line_points) if len(line_points) == 2 else None