import cv2
from typing import Tuple, List
from tracker import EuclideanDistTracker


def select_roi(frame: any) -> Tuple[int, int, int, int]:
    """
    Позволяет пользователю вручную выбрать ROI на первом кадре видео.

    :param frame: Первый кадр из видео.
    :return: Координаты ROI (x, y, width, height).
    """
    print("[INFO] Выберите ROI и нажмите ENTER или SPACE.")
    bbox = cv2.selectROI("Выбор ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Выбор ROI")
    return bbox


def detect_objects(roi_frame: any, detector: cv2.BackgroundSubtractor) -> List[List[int]]:
    """
    Выполняет детекцию объектов на основе фоновой разницы.

    :param roi_frame: Изображение ROI.
    :param detector: Объект фоново-вычитателя OpenCV.
    :return: Список координат обнаруженных объектов в формате [x, y, w, h].
    """
    mask = detector.apply(roi_frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    return detections, mask


def draw_tracks(roi_frame: any, tracks: List[List[int]]) -> None:
    """
    Отображает отслеживаемые объекты на ROI.

    :param roi_frame: Область интереса (ROI).
    :param tracks: Список треков, содержащих [x, y, w, h, id].
    """
    for x, y, w, h, object_id in tracks:
        cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(roi_frame, str(object_id), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def process_video(video_path: str) -> None:
    """
    Основная функция обработки видео: выбирает ROI, детектирует и трекает объекты.

    :param video_path: Путь к видеофайлу.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть видео.")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Не удалось прочитать первый кадр.")
        return

    bbox = select_roi(frame)
    x_roi, y_roi, w_roi, h_roi = bbox

    tracker = EuclideanDistTracker()
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

        detections, mask = detect_objects(roi, object_detector)
        tracked_objects = tracker.update(detections)
        draw_tracks(roi, tracked_objects)

        # Визуализация
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 2)
        cv2.imshow("ROI", roi)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video("video/highway.mp4")
