"""
Точка входа в систему обучения и оценки гомографии/аффинного отображения.
"""

from train import HPolyMapperTrainer
from predict import HPolyPredictor


def main() -> None:
    """
    Основная функция запуска.

    mode:
        - "train"  → обучение модели
        - "predict" → оценка модели на валидации
    """

    mode = "predict"  # "train" | "predict"

    split_path = "split.json"
    model_path = "artifacts/hpoly_mapper.pkl"

    if mode == "train":
        trainer = HPolyMapperTrainer()
        trainer.train(split_path)
        trainer.save(model_path)

        print("\nОбучение завершено. Модель сохранена.")

    elif mode == "predict":
        predictor = HPolyPredictor.load(model_path)
        predictor.run(split_path)

        print("\nОценка завершена. Метрики сохранены в artifacts/")

    else:
        raise ValueError("mode должен быть 'train' или 'predict'")


if __name__ == "__main__":
    main()
