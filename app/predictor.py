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

The API loads the model once at startup through `load(model_dir)`. Without
`model/model.joblib` it serves a keyword stand-in (version `stub-0.1`) so the
service can still be tested. A model file that exists but fails to load stops
the service from starting instead of falling back to the stand-in.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.textproc import build_input, clean_text  # noqa: E402

MODEL_DIR = ROOT / "model"
STUB_VERSION = "stub-0.1"

# Tickets the category head is unsure about go to a human queue.
# Emoji-only text landed near 0.31 on this model; the clearer samples sat above 0.50.
REVIEW_THRESHOLD = 0.40

# Imported for the unpickler. Do not remove.
_ = clean_text

logger = logging.getLogger(__name__)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _subjects_for(texts: list[str], subjects: list[str] | None) -> list[str]:
    if subjects is None:
        return [""] * len(texts)
    if len(subjects) != len(texts):
        raise ValueError("subjects and texts must be the same length")
    return subjects


class TicketModel:
    """The trained bundle in `model/`: category, secondary and urgent heads."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        import joblib
        import sklearn

        labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        fitted_with = str(labels.get("sklearn_version", ""))
        if fitted_with and fitted_with != sklearn.__version__:
            warnings.warn(
                f"model.joblib was fit with scikit-learn {fitted_with}; "
                f"this process has {sklearn.__version__}. "
                "Pin the same version in the container or the file may not load.",
                stacklevel=2,
            )
        bundle = joblib.load(model_dir / "model.joblib")
        self.model_version = str(labels["model_version"])
        self._team_of = labels["category_to_team"]
        self._cat = bundle["category"]
        self._sec = bundle["secondary"]
        self._urg = bundle["urgent"]

    def predict(self, texts: list[str], subjects: list[str] | None = None) -> list[dict]:
        """Classify tickets in order.

        `subjects` is optional and must be the same length as `texts` when given.
        An empty subject is correct for chat and call transcripts.
        """
        subjects = _subjects_for(texts, subjects)
        inputs = [build_input(subject, text) for subject, text in zip(subjects, texts)]
        if not inputs:
            return []

        proba = self._cat.predict_proba(inputs)
        cats = self._cat.classes_[proba.argmax(1)]
        confs = proba.max(1)
        sec_proba = self._sec.predict_proba(inputs)
        sec_classes = self._sec.classes_
        urgent_pred = self._urg.predict(inputs)

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
                "team": self._team_of[cat],
                "is_urgent": urgent,
                "confidence": confidence,
                "model_version": self.model_version,
                "needs_human_review": confidence < REVIEW_THRESHOLD,
            })
        return out


class StubModel:
    """Keyword stand-in, used only when model/model.joblib is absent."""

    model_version = STUB_VERSION

    # Earlier rules win ties; safety outranks everything, matching DATA_NOTES.md.
    _RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("safety_conduct", ("scared", "not stopping", "unsafe", "harass", "accident", "threat", "weapon")),
        ("delivery_delay", ("late", "delay", "not arrived", "has not arrived", "still waiting")),
        ("order_missing_wrong", ("missing", "wrong order", "wrong item", "not in the bag")),
        ("lost_item", ("left my", "forgot", "lost my", "wallet", "bag")),
        ("food_quality", ("cold", "spoiled", "stale", "undercooked", "hair in")),
        ("ride_trip_issue", ("wrong turn", "ride", "trip", "route")),
        ("account_promo", ("otp", "promo", "coupon", "login", "password")),
        ("app_technical", ("crash", "bug", "not loading", "error code")),
        ("payment_refund", ("refund", "charged", "payment", "double charge")),
        ("spam_irrelevant", ("lottery", "click here", "you won", "unsubscribe")),
    )
    _URGENT = ("scared", "emergency", "insulin", "diabetic", "bleeding", "accident", "fraud", "not stopping")

    def predict(self, texts: list[str], subjects: list[str] | None = None) -> list[dict]:
        subjects = _subjects_for(texts, subjects)
        return [self._one(build_input(s, t).lower()) for s, t in zip(subjects, texts)]

    def _one(self, text: str) -> dict:
        hits = [
            (sum(word in text for word in words), -rank, category)
            for rank, (category, words) in enumerate(self._RULES)
        ]
        hits = sorted((h for h in hits if h[0]), reverse=True)
        urgent = any(word in text for word in self._URGENT)
        if not hits:
            return {"category": "general_inquiry", "secondary_category": None, "is_urgent": urgent, "confidence": 0.3}
        category = hits[0][2]
        secondary = next((c for _, _, c in hits[1:] if c != "spam_irrelevant"), None)
        return {
            "category": category,
            "secondary_category": secondary,
            "is_urgent": urgent or category == "safety_conduct",
            "confidence": min(0.9, 0.35 + 0.15 * hits[0][0]),
        }


def load(model_dir: Path = MODEL_DIR) -> TicketModel | StubModel:
    """Load the trained model, or the stand-in when model.joblib is absent."""
    if not (model_dir / "model.joblib").exists():
        logger.warning("%s has no model.joblib; serving the keyword stand-in", model_dir)
        return StubModel()
    model = TicketModel(model_dir)
    model.predict(["warm-up"])
    logger.info("loaded model %s from %s", model.model_version, model_dir)
    return model


_default: TicketModel | None = None


def _model() -> TicketModel:
    global _default
    if _default is None:
        if not (MODEL_DIR / "model.joblib").exists():
            raise FileNotFoundError(f"Missing model file: {MODEL_DIR / 'model.joblib'}")
        _default = TicketModel(MODEL_DIR)
    return _default


def model_version() -> str:
    """Version string shared by /health, /predict and /predict/batch."""
    return _model().model_version


def predict(texts: list[str], subjects: list[str] | None = None) -> list[dict]:
    return _model().predict(texts, subjects)


if __name__ == "__main__":
    demo = predict(
        ["කාර් එකේ මගේ බෑග් එක අමතක වුණා, ඩ්‍රයිවර්ට කතා කරන්න පුළුවන්ද?"],
        [""],
    )
    print(model_version())
    for key, value in demo[0].items():
        print(f"{key}: {value}")
