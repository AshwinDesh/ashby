# relevant-priors

Simple API to score prior studies as relevant for each case.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn app.main:app --reload
```

The API exposes `POST /predict`.

For a production-style local run:

```bash
PORT=8000 uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
```

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

## Deploy

The app is deployable as either a Python web service or a Docker container. The saved model artifact at `models/model.joblib` must be included with the deployed code.

Python web service command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
```

Docker:

```bash
docker build -t relevant-priors .
docker run --rm -p 8000:8000 -e PORT=8000 relevant-priors
```

## Request shape

The endpoint accepts:

- `challenge_id` (string)
- `schema_version` (number)
- `generated_at` (ISO datetime)
- `cases` (list of case records with current/prior studies)

## Example call

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "challenge_id": "relevant-priors-v1",
    "schema_version": 1,
    "generated_at": "2026-04-16T12:00:00.000Z",
    "cases": [
      {
        "case_id": "1001016",
        "patient_id": "606707",
        "patient_name": "Andrews, Micheal",
        "current_study": {
          "study_id": "3100042",
          "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
          "study_date": "2026-03-08"
        },
        "prior_studies": [
          {
            "study_id": "2453245",
            "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
            "study_date": "2020-03-08"
          },
          {
            "study_id": "992654",
            "study_description": "CT HEAD WITHOUT CNTRST",
            "study_date": "2021-03-08"
          }
        ]
      }
    ]
  }'
```
