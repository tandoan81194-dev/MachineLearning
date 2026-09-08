from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import DATABASE_URL
from app.schemas import AnalysisRunInput, LeadInput, LeadPrediction, StoredEmployeeLeadScore


@dataclass(frozen=True)
class AnalysisPersistenceResult:
    analysis_run_id: str
    saved_to_database: bool
    status: str


@dataclass(frozen=True)
class EmployeeLeadHistoryResult:
    saved_to_database: bool
    status: str
    leads: list[StoredEmployeeLeadScore]


def _database_unavailable_result() -> EmployeeLeadHistoryResult:
    return EmployeeLeadHistoryResult(
        saved_to_database=False,
        status="DATABASE_URL is not configured; persisted employee history is unavailable.",
        leads=[],
    )


def _ensure_schema(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id UUID PRIMARY KEY,
            source_filename TEXT,
            source_sheet TEXT,
            source_s3_bucket TEXT,
            source_s3_key TEXT,
            source_s3_uri TEXT,
            source_s3_url TEXT,
            row_count INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    for column_name in [
        "source_s3_bucket",
        "source_s3_key",
        "source_s3_uri",
        "source_s3_url",
    ]:
        cursor.execute(
            f"ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS {column_name} TEXT"
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS employee_lead_scores (
            id UUID PRIMARY KEY,
            analysis_run_id UUID NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            source_row INTEGER,
            employee_id TEXT,
            employee_name TEXT,
            department TEXT,
            role TEXT,
            manager TEXT,
            scenario_note TEXT,
            lead_payload JSONB NOT NULL,
            employment_probability DOUBLE PRECISION NOT NULL,
            segment TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def save_analysis_run(
    payload: AnalysisRunInput,
    predictions: list[LeadPrediction],
    model_version: str,
) -> AnalysisPersistenceResult:
    analysis_run_id = str(uuid.uuid4())

    if not DATABASE_URL:
        return AnalysisPersistenceResult(
            analysis_run_id=analysis_run_id,
            saved_to_database=False,
            status="DATABASE_URL is not configured; analysis was not persisted.",
        )

    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError:
        return AnalysisPersistenceResult(
            analysis_run_id=analysis_run_id,
            saved_to_database=False,
            status="PostgreSQL driver is not installed. Install psycopg[binary] to enable persistence.",
        )

    try:
        with psycopg.connect(DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                _ensure_schema(cursor)
                cursor.execute(
                    """
                    INSERT INTO analysis_runs (
                        id,
                        source_filename,
                        source_sheet,
                        source_s3_bucket,
                        source_s3_key,
                        source_s3_uri,
                        source_s3_url,
                        row_count,
                        model_version
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        analysis_run_id,
                        payload.source_filename,
                        payload.source_sheet,
                        payload.source_s3_bucket,
                        payload.source_s3_key,
                        payload.source_s3_uri,
                        payload.source_s3_url,
                        len(payload.employees),
                        model_version,
                    ),
                )

                for employee, prediction in zip(payload.employees, predictions, strict=True):
                    cursor.execute(
                        """
                        INSERT INTO employee_lead_scores (
                            id,
                            analysis_run_id,
                            source_row,
                            employee_id,
                            employee_name,
                            department,
                            role,
                            manager,
                            scenario_note,
                            lead_payload,
                            employment_probability,
                            segment,
                            recommended_action,
                            model_version
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(uuid.uuid4()),
                            analysis_run_id,
                            employee.source_row,
                            employee.employee_id,
                            employee.employee_name,
                            employee.department,
                            employee.role,
                            employee.manager,
                            employee.scenario_note,
                            Jsonb(json.loads(employee.lead.model_dump_json())),
                            prediction.employment_probability,
                            prediction.segment,
                            prediction.recommended_action,
                            prediction.model_version,
                        ),
                    )

            connection.commit()
    except Exception as exc:
        return AnalysisPersistenceResult(
            analysis_run_id=analysis_run_id,
            saved_to_database=False,
            status=f"Database persistence failed: {exc}",
        )

    return AnalysisPersistenceResult(
        analysis_run_id=analysis_run_id,
        saved_to_database=True,
        status="Analysis run and employee predictions were persisted.",
    )


def load_employee_lead_scores(limit: int = 500) -> EmployeeLeadHistoryResult:
    if not DATABASE_URL:
        return _database_unavailable_result()

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        return EmployeeLeadHistoryResult(
            saved_to_database=False,
            status="PostgreSQL driver is not installed. Install psycopg[binary] to enable persistence.",
            leads=[],
        )

    try:
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as connection:
            with connection.cursor() as cursor:
                _ensure_schema(cursor)
                cursor.execute(
                    """
                    SELECT
                        scores.id::text AS id,
                        scores.analysis_run_id::text AS analysis_run_id,
                        runs.source_filename,
                        runs.source_sheet,
                        runs.source_s3_bucket,
                        runs.source_s3_key,
                        runs.source_s3_uri,
                        runs.source_s3_url,
                        scores.source_row,
                        scores.employee_id,
                        scores.employee_name,
                        scores.department,
                        scores.role,
                        scores.manager,
                        scores.scenario_note,
                        scores.lead_payload,
                        scores.employment_probability,
                        scores.segment,
                        scores.recommended_action,
                        scores.model_version,
                        scores.created_at
                    FROM employee_lead_scores AS scores
                    INNER JOIN analysis_runs AS runs
                        ON runs.id = scores.analysis_run_id
                    ORDER BY scores.created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
    except Exception as exc:
        return EmployeeLeadHistoryResult(
            saved_to_database=False,
            status=f"Database read failed: {exc}",
            leads=[],
        )

    leads = [_row_to_stored_lead_score(row) for row in rows]
    return EmployeeLeadHistoryResult(
        saved_to_database=True,
        status=f"Loaded {len(leads)} persisted employee lead scores.",
        leads=leads,
    )


def _row_to_stored_lead_score(row: dict[str, Any]) -> StoredEmployeeLeadScore:
    lead_payload = row["lead_payload"]
    if isinstance(lead_payload, str):
        lead_payload = json.loads(lead_payload)

    return StoredEmployeeLeadScore(
        id=row["id"],
        analysis_run_id=row["analysis_run_id"],
        source_filename=row["source_filename"],
        source_sheet=row["source_sheet"],
        source_s3_bucket=row["source_s3_bucket"],
        source_s3_key=row["source_s3_key"],
        source_s3_uri=row["source_s3_uri"],
        source_s3_url=row["source_s3_url"],
        source_row=row["source_row"],
        employee_id=row["employee_id"],
        employee_name=row["employee_name"],
        department=row["department"],
        role=row["role"],
        manager=row["manager"],
        scenario_note=row["scenario_note"],
        lead=LeadInput(**lead_payload),
        prediction=LeadPrediction(
            employment_probability=row["employment_probability"],
            segment=row["segment"],
            recommended_action=row["recommended_action"],
            model_version=row["model_version"],
        ),
        created_at=row["created_at"],
    )
