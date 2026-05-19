import cv2

roi_points = []
drawing = False

def on_mouse(event, x, y, flags, param):
    global roi_points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_points = [roi_points[0], (x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        drawing = False

def select_roi(frame):
    """El usuario arrastra un rectángulo sobre el frame para definir el ROI."""
    global roi_points, drawing
    roi_points = []
    drawing = False

    clone = frame.copy()
    cv2.namedWindow("Seleccionar ROI")
    cv2.setMouseCallback("Seleccionar ROI", on_mouse)

    print("Arrastrá un rectángulo para definir el área de corredores. Enter para confirmar.")

    while True:
        display = clone.copy()

        if len(roi_points) == 2:
            cv2.rectangle(display, roi_points[0], roi_points[1], (255, 165, 0), 2)

        cv2.imshow("Seleccionar ROI", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(roi_points) == 2:  # Enter
            break
        if key == 27:
            roi_points = []
            break

    cv2.destroyWindow("Seleccionar ROI")

    if len(roi_points) == 2:
        x1 = min(roi_points[0][0], roi_points[1][0])
        y1 = min(roi_points[0][1], roi_points[1][1])
        x2 = max(roi_points[0][0], roi_points[1][0])
        y2 = max(roi_points[0][1], roi_points[1][1])
        return (x1, y1, x2, y2)
    return None