# main.py

import cv2
from tracker import YOLOTracker
from utils import find_first_video


def main():
    folder = 'video'
    video_path = find_first_video(folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл: {video_path}")
        return

    tracker = YOLOTracker(roi_resized=(1280, 720))
    tracker.cap = cap
    tracker.load_model()

    # Выбор ROI один раз
    tracker.select_roi()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] != tracker.roi_resized[0] or frame.shape[0] != tracker.roi_resized[1]:
            frame = cv2.resize(frame, tracker.roi_resized)

        tracker.predict(frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
