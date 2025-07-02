# base_prediction.py

__all__ = ['BasePrediction', 'PredictionType']

from abc import ABC
from dataclasses import dataclass
from typing import TypeVar, Generic

PredictionType = TypeVar('PredictionType', bound='BasePrediction')

@dataclass
class BasePrediction(ABC, Generic[PredictionType]):
    """ Суперкласс предсказаний, возвращаемых нейронкой любого типа """
    cls_id: int
    conf: float
