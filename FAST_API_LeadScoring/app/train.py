from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import (
    EVALUATION_REPORT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    RANDOM_STATE,
    SCHEMA_PATH,
    TEST_SIZE,
)
from app.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def build_model_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    clean_dataset = dataset.dropna(subset=[TARGET_COLUMN]).copy()
    clean_dataset[TARGET_COLUMN] = clean_dataset[TARGET_COLUMN].astype(int)
    return clean_dataset[FEATURE_COLUMNS], clean_dataset[TARGET_COLUMN]


def evaluate_predictions(y_true: pd.Series, y_pred: list[int], y_proba: list[float]) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix,
        },
    }


def train_model(
    dataset: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    schema_path: Path = SCHEMA_PATH,
    report_path: Path = EVALUATION_REPORT_PATH,
) -> dict[str, Any]:
    X, y = split_features_target(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_pred.tolist(), y_proba.tolist())
    metrics.update(
        {
            "model_version": MODEL_VERSION,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_rate": float(y.mean()),
            "features": FEATURE_COLUMNS,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
        }
    )

    artifact = {
        "model_version": MODEL_VERSION,
        "pipeline": pipeline,
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    schema_path = Path(schema_path)
    report_path = Path(report_path)
    for path in [model_path.parent, metrics_path.parent, schema_path.parent, report_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    schema_path.write_text(json.dumps(build_schema(dataset), indent=2), encoding="utf-8")
    report_path.write_text(render_evaluation_report(metrics), encoding="utf-8")
    return metrics


def build_schema(dataset: pd.DataFrame) -> dict[str, Any]:
    categorical_values = {
        column: sorted(str(value) for value in dataset[column].dropna().unique())
        for column in CATEGORICAL_FEATURES
    }
    return {
        "model_version": MODEL_VERSION,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_values": categorical_values,
    }


def render_evaluation_report(metrics: dict[str, Any]) -> str:
    matrix = metrics["confusion_matrix"]["matrix"]
    return "\n".join(
        [
            "# Evaluation Report",
            "",
            f"Model version: `{metrics['model_version']}`",
            "",
            "## Metrics",
            "",
            f"- Accuracy: {metrics['accuracy']:.4f}",
            f"- Precision: {metrics['precision']:.4f}",
            f"- Recall: {metrics['recall']:.4f}",
            f"- F1: {metrics['f1']:.4f}",
            f"- ROC-AUC: {metrics['roc_auc']:.4f}",
            "",
            "## Data Split",
            "",
            f"- Train rows: {metrics['train_rows']}",
            f"- Test rows: {metrics['test_rows']}",
            f"- Positive rate: {metrics['positive_rate']:.4f}",
            "",
            "## Confusion Matrix",
            "",
            "| Actual / Predicted | 0 | 1 |",
            "|---|---:|---:|",
            f"| 0 | {matrix[0][0]} | {matrix[0][1]} |",
            f"| 1 | {matrix[1][0]} | {matrix[1][1]} |",
            "",
        ]
    )
