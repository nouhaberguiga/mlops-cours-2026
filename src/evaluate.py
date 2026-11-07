"""Évalue le modèle entraîné et sauvegarde les métriques pour DVC."""

import json
from pathlib import Path

import mlflow.sklearn
import yaml

from src.dataset_config import load_config
from src.train import compute_metrics, decision_threshold

TRAIN_CSV = Path("data/processed/train.csv")
TEST_CSV = Path("data/processed/test.csv")
MODEL_DIR = "models/model"
METRICS_FILE = Path("reports/metrics.json")


def main() -> None:
    cfg = load_config()

    params = yaml.safe_load(
        Path("params.yaml").read_text(encoding="utf-8")
    )

    train_params = params["train"]

    _, y_train = cfg.read_xy(TRAIN_CSV)
    X_test, y_test = cfg.read_xy(TEST_CSV)

    threshold = decision_threshold(
        cfg,
        train_params["class_weight"],
        y_train,
    )

    model = mlflow.sklearn.load_model(MODEL_DIR)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(
        cfg,
        y_test,
        probabilities,
        threshold,
    )

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Métriques enregistrées dans {METRICS_FILE}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
