from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EVALUATION_REPORT_PATH, METRICS_PATH, MODEL_PATH, PROCESSED_DATA_PATH, SCHEMA_PATH
from app.data import load_processed_dataset
from app.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline lead scoring model.")
    parser.add_argument("--data", default=str(PROCESSED_DATA_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    parser.add_argument("--schema", default=str(SCHEMA_PATH))
    parser.add_argument("--report", default=str(EVALUATION_REPORT_PATH))
    args = parser.parse_args()

    dataset = load_processed_dataset(Path(args.data))
    metrics = train_model(
        dataset,
        model_path=Path(args.model),
        metrics_path=Path(args.metrics),
        schema_path=Path(args.schema),
        report_path=Path(args.report),
    )

    print(f"Saved model: {Path(args.model).resolve()}")
    print(f"Saved metrics: {Path(args.metrics).resolve()}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
