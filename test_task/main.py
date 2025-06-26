from pathlib import Path

from tracker import YOLOTracker
import cv2

CURRENT_DIR = Path(__file__).resolve().parent

def main():
    """
    Точка входа в программу. Загружает видео, запускает трекинг объектов,
    выводит результат наличия транспорта перед шлагбаумом в реальном времени.
    """
    tracker = YOLOTracker()
    tracker.load_model()

    video_path = str(CURRENT_DIR / 'video/cvtest.avi')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracker.predict(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
