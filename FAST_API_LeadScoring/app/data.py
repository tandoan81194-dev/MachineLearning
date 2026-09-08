from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from app.config import DEFAULT_RAW_DATA_DIR, PROCESSED_DATA_PATH
from app.features import FEATURE_COLUMNS, ID_COLUMN, OUTPUT_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    columns: int
    duplicate_ids: int
    target_missing: int
    target_positive_rate: float
    missing_cells: dict[str, int]


def _read_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw Excel file: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw CSV file: {path}")
    return pd.read_csv(path)


def load_raw_sources(raw_data_dir: Path = DEFAULT_RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    raw_data_dir = Path(raw_data_dir)
    return {
        "enrollies": _read_excel(raw_data_dir / "Enrollies.xlsx", "enrollies"),
        "education": _read_excel(raw_data_dir / "enrollies_education.xlsx", "enrollies_education"),
        "work_experience": _read_csv(raw_data_dir / "work_experience.csv"),
        "training_hours": _read_csv(raw_data_dir / "training_hours.csv"),
        "city_dev_index": _read_csv(raw_data_dir / "city_dev_index.csv"),
        "employment": _read_csv(raw_data_dir / "employment.csv"),
    }


def prepare_dataset(raw_data_dir: Path = DEFAULT_RAW_DATA_DIR) -> pd.DataFrame:
    sources = load_raw_sources(raw_data_dir)

    enrollies = sources["enrollies"].rename(columns={"city": "city"})
    education = sources["education"]
    work = sources["work_experience"]
    training_hours = sources["training_hours"]
    employment = sources["employment"]

    city_dev = sources["city_dev_index"].rename(
        columns={
            "City": "city",
            "City Development Index": "city_development_index",
        }
    )
    city_dev = city_dev[["city", "city_development_index"]]

    dataset = (
        enrollies.merge(education, on=ID_COLUMN, how="left")
        .merge(work, on=ID_COLUMN, how="left")
        .merge(training_hours, on=ID_COLUMN, how="left")
        .merge(city_dev, on="city", how="left")
        .merge(employment, on=ID_COLUMN, how="inner")
    )

    dataset = dataset.drop(columns=["full_name"], errors="ignore")
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype("Int64")
    dataset = dataset[OUTPUT_COLUMNS]
    return dataset


def build_quality_report(dataset: pd.DataFrame) -> DataQualityReport:
    target = dataset[TARGET_COLUMN].dropna()
    positive_rate = float(target.mean()) if len(target) else 0.0
    missing_cells = {
        column: int(dataset[column].isna().sum())
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if int(dataset[column].isna().sum()) > 0
    }
    return DataQualityReport(
        rows=int(dataset.shape[0]),
        columns=int(dataset.shape[1]),
        duplicate_ids=int(dataset[ID_COLUMN].duplicated().sum()),
        target_missing=int(dataset[TARGET_COLUMN].isna().sum()),
        target_positive_rate=positive_rate,
        missing_cells=missing_cells,
    )


def save_dataset(
    output_path: Path = PROCESSED_DATA_PATH,
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR,
) -> tuple[pd.DataFrame, DataQualityReport]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(raw_data_dir=raw_data_dir)
    report = build_quality_report(dataset)
    dataset.to_csv(output_path, index=False)

    report_path = output_path.with_suffix(".quality.json")
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return dataset, report


def load_processed_dataset(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run prepare-data first."
        )
    return pd.read_csv(path)
