# prediction_handler.py

from typing import Generic, List
from src.predictions.base_prediction import PredictionType

class PredictionHandler(Generic[PredictionType]):
    """
    Универсальный обработчик для предсказаний любого типа (TrackerPrediction, DetectionPrediction и др.)
    """
    def __init__(self):
        self.predictions: List[PredictionType] = []

    def add(self, prediction: PredictionType) -> None:
        self.predictions.append(prediction)

    def get_last(self) -> PredictionType:
        return self.predictions[-1]

    def clear(self) -> None:
        self.predictions.clear()

    def __len__(self) -> int:
        return len(self.predictions)
