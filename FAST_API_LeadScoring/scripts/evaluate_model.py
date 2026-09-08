from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EVALUATION_REPORT_PATH, METRICS_PATH, MODEL_PATH, PROCESSED_DATA_PATH
from app.data import load_processed_dataset
from app.evaluate import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved model on the processed dataset.")
    parser.add_argument("--data", default=str(PROCESSED_DATA_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    parser.add_argument("--report", default=str(EVALUATION_REPORT_PATH))
    args = parser.parse_args()

    dataset = load_processed_dataset(Path(args.data))
    metrics = evaluate_model(
        dataset,
        model_path=Path(args.model),
        metrics_path=Path(args.metrics),
        report_path=Path(args.report),
    )

    print(f"Evaluated model: {Path(args.model).resolve()}")
    print(f"Evaluation rows: {metrics['evaluation_rows']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
