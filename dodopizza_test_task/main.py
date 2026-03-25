# main.py

import argparse
from ui import MainWindow
from yolo_detector import YOLODetector
from opencv_detector import OpenCVDetector

if __name__ == "__main__":
    """
    Точка входа в приложение для анализа видео.

    Возможны два способа запуска:
    1. Через UI: выбираем видео, тип детектора (YOLO/OpenCV) и опцию записи.
    2. Через командную строку:
        python main.py --video video1.mp4
       В этом случае используется YOLO, запись видео отключена.
    """

    # Создание парсера аргументов для CLI
    parser = argparse.ArgumentParser(description="Анализ видео столика")
    parser.add_argument("--video", type=str, help="Путь к видеофайлу для анализа через CLI")
    args = parser.parse_args()

    if args.video:
        # Запуск без UI через CLI, используем YOLO по умолчанию, запись видео отключена
        video_path = args.video
        print(f"[INFO] Анализ видео {video_path} через YOLO (record=False)")
        detector = YOLODetector(video_path, record=False)
        detector.run()
    else:
        # Обычный запуск через UI
        ui = MainWindow()
        video_path, detector_choice, record_video = ui.run()

        # Инициализация выбранного детектора
        if detector_choice == "YOLO":
            detector = YOLODetector(video_path, record=record_video)
        else:
            detector = OpenCVDetector(video_path, record=record_video)

        # Запуск основного цикла обработки видео
        detector.run()