# main.py

from ui import MainWindow
from detectors.yolo_detector import YOLODetector
from detectors.opencv_detector import OpenCVDetector

if __name__ == "__main__":
    ui = MainWindow()
    video_path = ui.run()

    # Выбор детектора:
    detector = YOLODetector(video_path)
    # detector = OpenCVDetector(video_path)

    detector.run()