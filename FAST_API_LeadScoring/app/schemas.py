from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class LeadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str = Field(..., examples=["city_103"])
    gender: str | None = Field(default=None, examples=["Male"])
    enrolled_university: str | None = Field(default=None, examples=["no_enrollment"])
    education_level: str | None = Field(default=None, examples=["Graduate"])
    major_discipline: str | None = Field(default=None, examples=["STEM"])
    relevent_experience: str | None = Field(default=None, examples=["Has relevent experience"])
    experience: str | None = Field(default=None, examples=[">20"])
    company_size: str | None = Field(default=None, examples=["50-99"])
    company_type: str | None = Field(default=None, examples=["Pvt Ltd"])
    last_new_job: str | None = Field(default=None, examples=["1"])
    training_hours: float = Field(..., ge=0, examples=[36])
    city_development_index: float | None = Field(default=None, ge=0, le=1, examples=[0.92])


class LeadPrediction(BaseModel):
    employment_probability: float
    segment: str
    recommended_action: str
    model_version: str

class BatchLeadInput(BaseModel):
    leads: list[LeadInput]


class BatchLeadPrediction(BaseModel):
    predictions: list[LeadPrediction]


class EmployeeLeadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employee_id: str | None = None
    employee_name: str | None = None
    department: str | None = None
    role: str | None = None
    manager: str | None = None
    scenario_note: str | None = None
    source_row: int | None = None
    lead: LeadInput


class AnalysisRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_filename: str | None = None
    source_sheet: str | None = None
    source_s3_bucket: str | None = None
    source_s3_key: str | None = None
    source_s3_uri: str | None = None
    source_s3_url: str | None = None
    employees: list[EmployeeLeadInput]


class EmployeeLeadAnalysisResult(EmployeeLeadInput):
    prediction: LeadPrediction


class AnalysisRunResponse(BaseModel):
    analysis_run_id: str
    saved_to_database: bool
    persistence_status: str
    source_s3_bucket: str | None = None
    source_s3_key: str | None = None
    source_s3_uri: str | None = None
    source_s3_url: str | None = None
    results: list[EmployeeLeadAnalysisResult]


class SourceFileUploadResponse(BaseModel):
    saved_to_s3: bool
    status: str
    bucket: str | None = None
    key: str | None = None
    uri: str | None = None
    url: str | None = None


class StoredEmployeeLeadScore(BaseModel):
    id: str
    analysis_run_id: str
    source_filename: str | None = None
    source_sheet: str | None = None
    source_s3_bucket: str | None = None
    source_s3_key: str | None = None
    source_s3_uri: str | None = None
    source_s3_url: str | None = None
    source_row: int | None = None
    employee_id: str | None = None
    employee_name: str | None = None
    department: str | None = None
    role: str | None = None
    manager: str | None = None
    scenario_note: str | None = None
    lead: LeadInput
    prediction: LeadPrediction
    created_at: datetime


class EmployeeLeadHistoryResponse(BaseModel):
    saved_to_database: bool
    persistence_status: str
    total: int
    leads: list[StoredEmployeeLeadScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class ModelInfoResponse(BaseModel):
    # Code here
    pass