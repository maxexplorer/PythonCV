from ui import MainWindow
from yolo_detector import YOLODetector
from opencv_detector import OpenCVDetector

if __name__ == "__main__":
    ui = MainWindow()
    video_path, detector_choice, record_video = ui.run()

    if detector_choice == "YOLO":
        detector = YOLODetector(video_path, record=record_video)
    else:
        detector = OpenCVDetector(video_path, record=record_video)

    detector.run()