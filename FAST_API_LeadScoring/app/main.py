from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, Header, HTTPException, Query, Request

from app.config import METRICS_PATH, SOURCE_FILE_MAX_BYTES
from app.inference import get_service
from app.persistence import load_employee_lead_scores, save_analysis_run
from app.schemas import (
    AnalysisRunInput,
    AnalysisRunResponse,
    BatchLeadInput,
    BatchLeadPrediction,
    EmployeeLeadAnalysisResult,
    EmployeeLeadHistoryResponse,
    HealthResponse,
    LeadInput,
    LeadPrediction,
    ModelInfoResponse,
)

app = FastAPI(
    title="Training Lead Scoring API")

#Write API
#Popular: Check health of server
#Requests: 
# GET: get info
# POST: send data to server -> NEWLY ADD
# PUT: update data on server -> UPDATE WHAT IS ALREADY ON SERVER
# DELETE: delete data on server -> DELETE WHAT IS ALREADY ON SERVER

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    return HealthResponse(
        status="ok",
        model_loaded = True if service.model_path else False, #if model path is not None, then model is loaded
        model_version=service.model_version)

# API for users to predict one sample
# What kind of request?
# Response Model? => What needs to be returned to the user? 
# What function is being called? => How to call to that function? => What is the input and output of that function?

@app.post("/predict", response_model=LeadPrediction)
def predict(lead: LeadInput) -> LeadPrediction:
    service = get_service()
    prediction = service.predict_one(lead)
    return prediction

# Predict many leads at once
@app.post("/batch-predict", response_model=BatchLeadPrediction)
def batch_predict(payload: BatchLeadInput) -> BatchLeadPrediction:
    service = get_service()
    predictions = service.predict_many(payload.leads)
    return BatchLeadPrediction(predictions=predictions)