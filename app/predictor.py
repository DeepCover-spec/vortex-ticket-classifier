"""Serving entry point for the ticket model.

P4 calls `predict`. Pass ticket bodies and, for emails, the subject lines.
The dicts that come back are the model fields of the API response. P4 still
adds `ticket_id` and enforces the API key.

    from app.predictor import predict, model_version

    rows = predict(texts, subjects)

Team is never predicted. It is looked up from the fixed table in
`model/labels.json`. `spam_irrelevant` is forced to `is_urgent=False` and
`secondary_category=None`, which is what the response schema requires.

The pickled pipelines call `app.textproc.clean_text` by name, so this module
imports that function before loading the file.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.textproc import build_input, clean_text

MODEL_PATH = ROOT / "model" / "model.joblib"
LABELS_PATH = ROOT / "model" / "labels.json"

# Tickets the category head is unsure about go to a human queue.
# Emoji-only text landed near 0.31 on this model; the clearer samples sat above 0.50.
REVIEW_THRESHOLD = 0.40

# Imported for the unpickler. Do not remove.
_ = clean_text

_bundle: dict | None = None
_labels: dict | None = None


def _load() -> tuple[dict, dict]:
    global _bundle, _labels
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
        _labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        fitted_with = str(_labels.get("sklearn_version", ""))
        if fitted_with and fitted_with != sklearn.__version__:
            warnings.warn(
                f"model.joblib was fit with scikit-learn {fitted_with}; "
                f"this process has {sklearn.__version__}. "
                "Pin the same version in the container or the file may not load.",
                stacklevel=2,
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle, _labels


def model_version() -> str:
    """Version string shared by /health, /predict and /predict/batch."""
    _, labels = _load()
    return str(labels["model_version"])


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def predict(texts: list[str], subjects: list[str] | None = None) -> list[dict]:
    """Classify tickets in order.

    `subjects` is optional and must be the same length as `texts` when given.
    An empty subject is correct for chat and call transcripts.
    """
    if subjects is None:
        subjects = [""] * len(texts)
    if len(subjects) != len(texts):
        raise ValueError("subjects and texts must be the same length")

    bundle, labels = _load()
    team_of = labels["category_to_team"]
    version = str(labels["model_version"])
    cat_model = bundle["category"]
    sec_model = bundle["secondary"]
    urg_model = bundle["urgent"]

    inputs = [build_input(subject, text) for subject, text in zip(subjects, texts)]
    if not inputs:
        return []

    proba = cat_model.predict_proba(inputs)
    classes = cat_model.classes_
    cats = classes[proba.argmax(1)]
    confs = proba.max(1)

    sec_proba = sec_model.predict_proba(inputs)
    sec_classes = sec_model.classes_
    urgent_pred = urg_model.predict(inputs)

    out = []
    for i, cat in enumerate(cats):
        cat = str(cat)
        sec = None
        top = str(sec_classes[sec_proba[i].argmax()])
        if top != "none" and top != cat:
            sec = top
        urgent = _as_bool(urgent_pred[i])
        if cat == "spam_irrelevant":
            sec, urgent = None, False
        confidence = round(float(min(1.0, max(0.0, confs[i]))), 4)
        out.append({
            "category": cat,
            "secondary_category": sec,
            "team": team_of[cat],
            "is_urgent": urgent,
            "confidence": confidence,
            "model_version": version,
            "needs_human_review": confidence < REVIEW_THRESHOLD,
        })
    return out


if __name__ == "__main__":
    demo = predict(
        ["කාර් එකේ මගේ බෑග් එක අමතක වුණා, ඩ්‍රයිවර්ට කතා කරන්න පුළුවන්ද?"],
        [""],
    )
    print(model_version())
    for key, value in demo[0].items():
        print(f"{key}: {value}")
