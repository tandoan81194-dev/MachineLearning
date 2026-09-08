from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import MODEL_PATH
from app.features import FEATURE_COLUMNS
from app.schemas import LeadInput, LeadPrediction


class LeadScoringService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}. Run train first.")

        artifact = joblib.load(model_path)
        self.model_path = model_path
        self.model_version: str = artifact["model_version"]
        self.pipeline = artifact["pipeline"]
        self.features: list[str] = artifact["features"]
        self.categorical_features: list[str] = artifact["categorical_features"]
        self.numeric_features: list[str] = artifact["numeric_features"]

    def predict_one(self, lead: LeadInput | dict[str, Any]) -> LeadPrediction:
        if isinstance(lead, LeadInput):
            payload = lead.model_dump()
        else:
            payload = LeadInput(**lead).model_dump()

        row = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        return LeadPrediction(
            employment_probability=round(probability, 6),
            segment=segment_from_probability(probability),
            recommended_action=action_from_probability(probability),
            model_version=self.model_version,
        )

    def predict_many(self, leads: list[LeadInput]) -> list[LeadPrediction]:
        return [self.predict_one(lead) for lead in leads]

    def model_info(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "features": self.features,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
        }


def segment_from_probability(probability: float) -> str:
    if probability >= 0.7:
        return "high_potential"
    if probability >= 0.4:
        return "medium_potential"
    return "needs_nurturing"


def action_from_probability(probability: float) -> str:
    if probability >= 0.7:
        return "Assign senior counselor and offer advanced career track"
    if probability >= 0.4:
        return "Send targeted learning plan and schedule follow-up"
    return "Enroll in foundation support journey before sales escalation"


@lru_cache(maxsize=1)
def get_service() -> LeadScoringService:
    return LeadScoringService()
