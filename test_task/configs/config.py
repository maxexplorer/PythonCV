from pathlib import Path
from enum import Enum


CURRENT_DIR = Path(__file__).resolve().parent.parent


class Config:
    class Device(Enum):
        NONE = None
        CUDA = 'cuda'
        CPU = 'cpu'

    class TrackerConfig:
        # Конфигурация YOLO трекера
        enable = True
        model_path = str(CURRENT_DIR / 'models/yolo11n.pt')
        conf = 0.7
        verbose = False
        track_path = str(CURRENT_DIR / 'trackers/botsort.yaml')
        persist = True
