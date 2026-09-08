from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import (
    DEFAULT_RAW_DATA_DIR,
    EVALUATION_REPORT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    SCHEMA_PATH,
)
from app.data import load_processed_dataset, save_dataset
from app.evaluate import evaluate_model
from app.inference import LeadScoringService
from app.train import train_model


def command_prepare_data(args: argparse.Namespace) -> None:
    dataset, report = save_dataset(
        output_path=Path(args.output),
        raw_data_dir=Path(args.raw_data_dir),
    )
    print(f"Saved processed dataset: {Path(args.output).resolve()}")
    print(f"Rows: {report.rows}")
    print(f"Columns: {report.columns}")
    print(f"Duplicate IDs: {report.duplicate_ids}")
    print(f"Target positive rate: {report.target_positive_rate:.4f}")
    if report.missing_cells:
        print("Missing cells:")
        for column, count in report.missing_cells.items():
            print(f"  - {column}: {count}")
    print(f"Preview rows: {min(5, len(dataset))}")


def command_train(args: argparse.Namespace) -> None:
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


def command_evaluate(args: argparse.Namespace) -> None:
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


def command_predict(args: argparse.Namespace) -> None:
    payload = _load_payload(args.input_json, args.inline_json)
    service = LeadScoringService(Path(args.model))
    prediction = service.predict_one(payload)
    print(prediction.model_dump_json(indent=2))


def command_run_api(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Missing API dependencies. Install with: python -m pip install -r requirements.txt"
        ) from exc

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def _load_payload(input_json: str | None, inline_json: str | None) -> dict[str, Any]:
    if input_json:
        return json.loads(Path(input_json).read_text(encoding="utf-8"))
    if inline_json:
        return json.loads(inline_json)
    raise SystemExit("Provide --input-json path or --json payload.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lead-scoring",
        description="Python CLI for the Training Lead Scoring project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Merge raw Excel/CSV files into a processed ML dataset.")
    prepare.add_argument("--raw-data-dir", default=str(DEFAULT_RAW_DATA_DIR))
    prepare.add_argument("--output", default=str(PROCESSED_DATA_PATH))
    prepare.set_defaults(func=command_prepare_data)

    train = subparsers.add_parser("train", help="Train baseline lead scoring model.")
    train.add_argument("--data", default=str(PROCESSED_DATA_PATH))
    train.add_argument("--model", default=str(MODEL_PATH))
    train.add_argument("--metrics", default=str(METRICS_PATH))
    train.add_argument("--schema", default=str(SCHEMA_PATH))
    train.add_argument("--report", default=str(EVALUATION_REPORT_PATH))
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate the saved model on the processed dataset.")
    evaluate.add_argument("--data", default=str(PROCESSED_DATA_PATH))
    evaluate.add_argument("--model", default=str(MODEL_PATH))
    evaluate.add_argument("--metrics", default=str(METRICS_PATH))
    evaluate.add_argument("--report", default=str(EVALUATION_REPORT_PATH))
    evaluate.set_defaults(func=command_evaluate)

    predict = subparsers.add_parser("predict", help="Run local prediction from JSON input.")
    predict.add_argument("--model", default=str(MODEL_PATH))
    predict.add_argument("--input-json")
    predict.add_argument("--json", dest="inline_json")
    predict.set_defaults(func=command_predict)

    run_api = subparsers.add_parser("run-api", help="Run FastAPI service with uvicorn.")
    run_api.add_argument("--host", default="127.0.0.1")
    run_api.add_argument("--port", type=int, default=8001)
    run_api.add_argument("--reload", action="store_true")
    run_api.set_defaults(func=command_run_api)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
