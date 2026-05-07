# tracker_prediction.py

__all__ = ['TrackerPrediction']

from dataclasses import dataclass
from test_task.predictions.base_prediction import BasePrediction

@dataclass
class TrackerPrediction(BasePrediction['TrackerPrediction']):
    """ Предсказания, возвращаемые трекером YOLO """
    box: list[int]
    cls_name: str
    track_id: int
