# Vortex Ticket Classifier

TensorForge 2.0 Phase 2: a support-ticket classification API for RideEat, built to the organizers' OpenAPI spec v2.1.0 (`spec/`).

## Live API

**Base URL: https://vortex-ticket-classifier.up.railway.app**

- Health check (public): https://vortex-ticket-classifier.up.railway.app/health
- Interactive docs: https://vortex-ticket-classifier.up.railway.app/docs

Every endpoint except `/health` needs the API key, sent either as `X-API-Key: <key>` or as `Authorization: Bearer <key>`. The service reads the expected key from the `API_KEY` environment variable on the host. It is never stored in the repo or the image. If `API_KEY` is unset, the protected endpoints return `401` to everyone.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and model version |
| POST | `/predict` | Classify one ticket |
| POST | `/predict/batch` | Classify 1–100 tickets synchronously |
| POST | `/batch/jobs` | Submit 1–5000 tickets as a background job (returns `202`) |
| GET | `/batch/jobs/{job_id}` | Job status (`Retry-After` while queued or running) |
| GET | `/batch/jobs/{job_id}/results` | Paged results (`offset`, `limit`) |
| DELETE | `/batch/jobs/{job_id}` | Cancel or delete a job (`204`) |

Request body limits: 1 MiB for `/predict`, 5 MiB for `/predict/batch`, 25 MiB for `/batch/jobs`.

## Model

The served model is `model/model.joblib`, with its metadata in `model/labels.json` (current version `p3-tfidf-e5900389`, scikit-learn 1.9.0). Training code and the experiment log are in `training/`. If `model/model.joblib` is missing, the API falls back to a keyword stub (`stub-0.1`).

## Run locally

Requires Python 3.11.

```bash
pip install -r requirements-dev.txt
export API_KEY=dev-key            # PowerShell: $env:API_KEY = "dev-key"
uvicorn app.main:app --port 8000 --workers 1
```

With Docker:

```bash
docker build -t ticket-classifier .
docker run -p 8000:8000 -e API_KEY=dev-key ticket-classifier
```

The container needs no internet access at runtime.

## Tests

```bash
pytest                                   # in-process, uses TF_API_KEY (default: test-key)
TF_BASE_URL=http://localhost:8000 TF_API_KEY=dev-key pytest   # against a running server
```

To test the hosted service, set `TF_BASE_URL` to the base URL above and `TF_API_KEY` to the team key from your own environment. `test_job_oversize_body_is_413` uploads about 26 MB over a 60 s client timeout, so on a slow uplink it can time out before the `413` arrives.

## Hosting

- **Host:** Railway, Southeast Asia region, built from this repo's `Dockerfile`.
- **Process:** one replica, one uvicorn worker. The batch job queue lives in that one process.
- **Job store:** SQLite in `/app/data`, on a Railway volume, so finished jobs survive redeploys for their 6-hour retention. Jobs that were queued or running when the service restarts are marked failed (interrupted).
- **Secrets:** `API_KEY` is set only in the host's environment variables.
- **Uptime:** `.github/workflows/keepalive.yml` checks `/health` every 5 minutes.
