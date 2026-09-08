from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience for local dev
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

if load_dotenv:
    load_dotenv(WORKSPACE_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def find_workspace_data_dir() -> Path:
    for parent in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        candidate = parent / "training-course-data"
        if candidate.exists():
            return candidate
    return WORKSPACE_ROOT / "training-course-data"


def build_database_url() -> str | None:
    direct_url = os.getenv("DATABASE_URL")
    if direct_url:
        return direct_url

    host = os.getenv("DB_HOST")
    database = os.getenv("DB_DATABASE")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    port = os.getenv("DB_PORT", "5432")
    sslmode = os.getenv("DB_SSLMODE")

    if not all([host, database, user, password]):
        return None

    database_url = (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(database, safe='')}"
    )
    if sslmode:
        database_url = f"{database_url}?sslmode={quote(sslmode, safe='')}"
    return database_url


DEFAULT_RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", find_workspace_data_dir()))
DATABASE_URL = build_database_url()
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
AWS_S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET_NAME")
AWS_S3_PREFIX = os.getenv("S3_PREFIX", "lead-scoring/source-files").strip("/")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL") or None
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL") or None
AWS_S3_SERVER_SIDE_ENCRYPTION = os.getenv("AWS_S3_SERVER_SIDE_ENCRYPTION") or None
SOURCE_FILE_MAX_BYTES = int(os.getenv("SOURCE_FILE_MAX_BYTES", str(20 * 1024 * 1024)))
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "lead_scoring_dataset.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
SCHEMA_PATH = MODELS_DIR / "schema.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"

MODEL_VERSION = "baseline-2026-06"
RANDOM_STATE = 42
TEST_SIZE = 0.2
