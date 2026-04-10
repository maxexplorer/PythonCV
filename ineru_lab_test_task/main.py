# main.py

from train import HPolyMapperTrainer
from predict import HPolyPredictor


def main():
    mode = "predict"  # "train" | "predict"

    split_path = "split.json"
    model_path = "artifacts/hpoly_mapper.pkl"

    if mode == "train":
        trainer = HPolyMapperTrainer()
        trainer.train(split_path)
        trainer.save(model_path)

        print("\nTraining finished. Model saved.")

    elif mode == "predict":
        predictor = HPolyPredictor.load(model_path)
        predictor.run(split_path)

        print("\nEvaluation finished. Metrics saved to artifacts/")

    else:
        raise ValueError("mode must be 'train' or 'predict'")


if __name__ == "__main__":
    main()