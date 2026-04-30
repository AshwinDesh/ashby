# Deployment

## Requirements

- Include `models/model.joblib` in the deployed artifact.
- Install dependencies from `requirements.txt`.
- Start the API with a command that binds to `0.0.0.0` and the host-provided `PORT`.
- Submit the `/predict` URL, not the root URL.

## Recommended Render URL

Use `challenger-5494` as the Render service name. Render service names become part of the public `onrender.com` hostname, so the expected endpoint is:

```text
https://challenger-5494.onrender.com/predict
```

The shorter `https://challenger-5494/predict` is not a public internet URL unless you own and configure a real DNS domain.

## Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
```

## Docker

```bash
docker build -t relevant-priors .
docker run --rm -p 8000:8000 -e PORT=8000 relevant-priors
```

Then verify:

```bash
curl "http://127.0.0.1:8000/health"
```

## Submission Zip

Create a clean archive from the repository root:

```bash
zip -r relevant-priors-submission.zip \
  app models Dockerfile Procfile requirements.txt README.md DEPLOYMENT.md \
  experiments.md analyze_errors.py evaluate_public.py
```

Do not include `.venv/`, `__pycache__/`, `.synq/`, or the public eval JSON in the deployment image. The public eval JSON is useful locally, but the hosted endpoint only needs source code, dependencies, and the saved model artifact.
