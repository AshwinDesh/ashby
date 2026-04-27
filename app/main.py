from __future__ import annotations

from datetime import date, datetime

from fastapi import FastAPI
from pydantic import BaseModel

from app.model import RelevantPriorModel, StudyComparison

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
    prediction_metadata: list[tuple[str, str]] = []
    comparisons: list[StudyComparison] = []

    for case in payload.cases:
        for prior in case.prior_studies:
            prediction_metadata.append((case.case_id, prior.study_id))
            comparisons.append(
                StudyComparison(
                    current_study_description=case.current_study.study_description,
                    current_study_date=case.current_study.study_date,
                    prior_study_description=prior.study_description,
                    prior_study_date=prior.study_date,
                )
            )

    for (case_id, study_id), is_relevant in zip(
        prediction_metadata, model.predict_is_relevant_many(comparisons), strict=True
    ):
        predictions.append(
            Prediction(
                case_id=case_id,
                study_id=study_id,
                predicted_is_relevant=is_relevant,
            )
        )

    return PredictionResponse(predictions=predictions)
