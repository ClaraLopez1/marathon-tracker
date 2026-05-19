import cv2
import sys
import os

VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "video.mp4")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: no se pudo abrir el video '{VIDEO_PATH}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video abierto. {width}x{height} | FPS: {fps:.1f} | Frames totales: {total_frames}")
    print("Presioná 'q' para salir.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fin del video.")
            break

        cv2.imshow("Marathon Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()