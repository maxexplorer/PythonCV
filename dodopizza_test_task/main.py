# main.py

from ui import MainWindow
from yolo_detector import YOLODetector
from opencv_detector import OpenCVDetector

if __name__ == "__main__":
    """
    Точка входа в приложение для анализа видео.
    UI позволяет выбрать видеофайл, тип детектора (YOLO или OpenCV) 
    и включить запись выходного видео.
    """

    # Создание графического окна для выбора видео, детектора и опции записи
    ui = MainWindow()

    # Запуск UI и получение выбранных пользователем параметров:
    # video_path — путь к видеофайлу
    # detector_choice — строка "YOLO" или "OpenCV"
    # record_video — булево значение, нужно ли записывать выходное видео
    video_path, detector_choice, record_video = ui.run()

    # Инициализация выбранного детектора с передачей пути к видео и параметра записи
    if detector_choice == "YOLO":
        detector = YOLODetector(video_path, record=record_video)
    else:
        detector = OpenCVDetector(video_path, record=record_video)

    # Запуск основного цикла обработки видео
    detector.run()