import cv2
import pandas as pd
import numpy as np
from typing import Tuple


def select_roi(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Позволяет пользователю выбрать ROI на кадре.
    """
    print("[INFO] Выберите ROI и нажмите ENTER или SPACE.")
    bbox = cv2.selectROI("Выбор ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Выбор ROI")
    return bbox


class TableStateTracker:
    """
    FSM для отслеживания состояний столика: Empty -> Full -> Empty
    """

    def __init__(self) -> None:
        self.current_state: str = "Empty"
        self.events: list[dict] = []

    def update(self, is_person_present: bool, timestamp: float) -> None:
        new_state = "Full" if is_person_present else "Empty"
        if new_state == self.current_state:
            return

        if self.current_state == "Empty" and new_state == "Full":
            self.events.append({"timestamp": round(timestamp, 3), "event": "Empty_to_Full"})

        elif self.current_state == "Full" and new_state == "Empty":
            self.events.append({"timestamp": round(timestamp, 3), "event": "Full_to_Empty"})

        self.events.append({"timestamp": round(timestamp, 3),
                            "event": f"{self.current_state}_to_{new_state}"})
        self.current_state = new_state

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)


class Analytics:
    """
    Метрики для анализа событий.
    """

    @staticmethod
    def compute_mean_idle_time(df: pd.DataFrame) -> float:
        times: list[float] = []
        last_empty_time: float | None = None

        for _, row in df.iterrows():
            if row["event"] == "Full_to_Empty":
                last_empty_time = row["timestamp"]
            elif row["event"] == "Empty_to_Full" and last_empty_time:
                times.append(row["timestamp"] - last_empty_time)
                last_empty_time = None

        return float(np.mean(times)) if times else 0.0


def draw_table_box(frame: np.ndarray, roi: tuple[int, int, int, int], state: str) -> None:
    """
    Рисует ROI столика на кадре с цветом в зависимости от состояния.
    """
    x, y, w, h = roi
    color = (0, 255, 0) if state == "Empty" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(frame, state, (x, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)