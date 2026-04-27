from __future__ import annotations

from datetime import date, datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import RelevantPriorModel

app = FastAPI(title="Relevant Priors API", version="1.0.0")
model = RelevantPriorModel()


class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: date


class Case(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    current_study: Study
    prior_studies: list[Study]


class PredictionRequest(BaseModel):
    challenge_id: str
    schema_version: int
    generated_at: datetime
    cases: list[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictionResponse(BaseModel):
    predictions: list[Prediction]


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    predictions: list[Prediction] = []

    for case in payload.cases:
        for prior in case.prior_studies:
            is_relevant = model.predict_is_relevant(
                current_study_description=case.current_study.study_description,
                current_study_date=case.current_study.study_date,
                prior_study_description=prior.study_description,
                prior_study_date=prior.study_date,
            )
            if is_relevant:
                predictions.append(
                    Prediction(
                        case_id=case.case_id,
                        study_id=prior.study_id,
                        predicted_is_relevant=True,
                    )
                )

    return PredictionResponse(predictions=predictions)
