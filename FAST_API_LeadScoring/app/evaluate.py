from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import EVALUATION_REPORT_PATH, METRICS_PATH, MODEL_PATH
from app.features import FEATURE_COLUMNS
from app.train import evaluate_predictions, split_features_target


def evaluate_model(
    dataset: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    report_path: Path = EVALUATION_REPORT_PATH,
) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run train first.")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]

    X, y = split_features_target(dataset)
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    metrics = evaluate_predictions(y, y_pred.tolist(), y_proba.tolist())
    metrics.update(
        {
            "model_version": artifact["model_version"],
            "evaluation_rows": int(len(X)),
            "features": FEATURE_COLUMNS,
        }
    )

    metrics_path = Path(metrics_path)
    report_path = Path(report_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
