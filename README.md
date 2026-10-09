# Vortex Ticket Classifier

TensorForge 2.0 Phase 2. This is the RideEat support ticket API. You send a ticket, it returns a category, an optional second category, the team, whether it is urgent, and a confidence.

The contract is the OpenAPI file in `spec/` (v2.1.0). The classifier is a model we trained (TF-IDF and a linear SVM). It does not call an LLM. The team is not predicted. It is looked up from the fixed category-to-team table.

## Links

- API: https://vortex-ticket-classifier.up.railway.app
- Health (no key): https://vortex-ticket-classifier.up.railway.app/health
- Demo: https://vortex-two-mu.vercel.app

## Run with Docker

This is the command to start it. The image already has the model in it, so it does not need internet after the build.

```bash
docker build -t ticket-classifier .
docker run -p 8000:8000 -e API_KEY=your-key ticket-classifier
```

It listens on port 8000. There is one worker on purpose. Batch jobs are stored in a single SQLite file, so more than one worker would split them.

## Run locally

Python 3.11.

```bash
pip install -r requirements-dev.txt
```

PowerShell:

```powershell
$env:API_KEY = "dev-key"
uvicorn app.main:app --port 8000 --workers 1
```

bash:

```bash
export API_KEY=dev-key
uvicorn app.main:app --port 8000 --workers 1
```

If `API_KEY` is not set, `/health` still works and every other endpoint returns 401. Send the key as `X-API-Key` or as `Authorization: Bearer <key>`. Do not commit the real key. It only belongs in the environment.

Interactive docs are at `/docs` once the server is up.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Public. Status and model version. |
| POST | `/predict` | One ticket. |
| POST | `/predict/batch` | 1 to 100 tickets, in the request. |
| POST | `/batch/jobs` | 1 to 5000 tickets. Returns 202 and a job id. |
| GET | `/batch/jobs/{job_id}` | Status. `Retry-After` is set while it is queued or running. |
| GET | `/batch/jobs/{job_id}/results` | Results, with `offset` and `limit`. 409 if the job is not finished. |
| DELETE | `/batch/jobs/{job_id}` | Drops the job. 204. |

Body size limits are 1 MB, 5 MB and 25 MB for `/predict`, `/predict/batch` and `/batch/jobs`.

## Model

The file the API loads is `model/model.joblib`. Version in `model/labels.json` is `p3-tfidf-e5900389`, fit with scikit-learn 1.9.0. The container has to use that same scikit-learn version or the file will not load.

Label comes from word unigrams plus character n-grams (2 to 5) and a LinearSVC. Confidence comes from a calibrated copy of that head, not from the label head, because calibration was changing the predicted class. A second category is only returned when P(none) is below 0.65. Urgency is a separate logistic regression. `training/experiments.md` has the runs we kept and the ones we dropped.

The number we report is validation macro-F1 0.6888 (accuracy 0.6613), fit on `train.csv` and scored on `validation.csv`. The served baseline before that was 0.6731. After that score was written down, `train_p3.py` refit the same model on train plus validation and saved `model/model.joblib`. Do not score that file on `validation.csv` and treat it as a new result.

To train again, put `train.csv` and `validation.csv` in `data/` (that folder is not in git) and run:

```bash
pip install -r requirements.txt
python training/train_p3.py
```

That overwrites `model/model.joblib`, `model/labels.json`, and the charts in `training/outputs/`.

If `model.joblib` is missing, the API still starts and serves a keyword stub called `stub-0.1`. If the file is there but broken, startup fails instead of falling back.

## Tests

```bash
pytest
```

That runs in-process and uses `TF_API_KEY` (default `test-key`).

Against a server that is already running:

```powershell
$env:TF_BASE_URL = "http://localhost:8000"
$env:TF_API_KEY = "dev-key"
pytest
```

`test_job_oversize_body_is_413` uploads about 26 MB. On a slow connection the client can time out before the 413 gets back.

## Hosting

Railway, Southeast Asia, built from the Dockerfile in this repo. One replica. Job rows are in `/app/data` on a volume, kept for 6 hours. If the process restarts while a job is queued or running, that job comes back as failed with code `interrupted`. `.github/workflows/keepalive.yml` calls `/health` every 5 minutes.
