"""Model plug-in point (P2 hands over here).

Interface agreed in the team plan::

    predict(texts: list[str]) -> list[dict]

Each dict has ``category`` and ``confidence`` (0 to 1). It may also carry
``secondary_category`` (a category or None) and ``is_urgent`` (bool). ``team``
is never predicted; the API looks it up from the fixed table.

``ticket_text`` decides how one ticket becomes the string the model sees. Train
on exactly the same function, or predictions will drift.

Drop the trained artifact at ``model/model.joblib``. Accepted shapes:

1. A dict::

       {
           "model_version": "v1",                 # shown in /health and every response
           "category_model": <estimator>,         # required, predict(list[str])
           "secondary_model": <estimator>,        # optional, labels or "none"
           "urgent_model": <estimator>,           # optional, 0/1 or bool
       }

2. A bare estimator with ``predict(list[str])``. Only the category comes from
   the model then. Urgency defaults to true for safety_conduct.

Train with the scikit-learn, numpy and scipy versions pinned in requirements.txt
or the file may not load inside the container.

Without model/model.joblib a keyword stand-in answers so the API, Docker image
and batch jobs can be tested. The stand-in is never the submitted model.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

STUB_VERSION = "stub-0.1"
_NO_LABEL = {"", "none", "null", "nan"}


def ticket_text(subject: str, text: str) -> str:
    subject = (subject or "").strip()
    return f"{subject}\n{text}" if subject else text


class StubModel:
    """Keyword stand-in used until a trained model is dropped in."""

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

    def predict(self, texts: list[str]) -> list[dict]:
        return [self._one(text.lower()) for text in texts]

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


class SklearnModel:
    def __init__(self, bundle: object, model_version: str):
        self.model_version = model_version
        if isinstance(bundle, dict):
            self.category_model = bundle.get("category_model") or bundle.get("model")
            self.secondary_model = bundle.get("secondary_model")
            self.urgent_model = bundle.get("urgent_model")
        else:
            self.category_model, self.secondary_model, self.urgent_model = bundle, None, None
        if not hasattr(self.category_model, "predict"):
            raise ValueError("model.joblib has no category model with predict()")

    def predict(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        categories = [str(label) for label in self.category_model.predict(texts)]
        confidences = _confidence(self.category_model, texts, categories)
        secondaries = _labels(self.secondary_model, texts)
        urgent = _flags(self.urgent_model, texts)
        return [
            {
                "category": category,
                "secondary_category": secondaries[i] if secondaries else None,
                "is_urgent": urgent[i] if urgent else category == "safety_conduct",
                "confidence": confidences[i],
            }
            for i, category in enumerate(categories)
        ]


def _confidence(model: object, texts: list[str], predicted: list[str]) -> list[float]:
    """Probability of the predicted class. Falls back to a softmax of SVM-style scores."""
    classes = [str(c) for c in getattr(model, "classes_", [])]
    if hasattr(model, "predict_proba"):
        rows = model.predict_proba(texts)
        return [_pick(row, classes, label) for row, label in zip(rows, predicted)]
    if hasattr(model, "decision_function"):
        rows = model.decision_function(texts)
        out = []
        for row, label in zip(rows, predicted):
            scores = [float(s) for s in (row if hasattr(row, "__len__") else [-row, row])]
            top = max(scores)
            exps = [math.exp(s - top) for s in scores]
            total = sum(exps)
            out.append(_pick([e / total for e in exps], classes, label))
        return out
    return [0.5] * len(texts)


def _pick(row, classes: list[str], label: str) -> float:
    values = [float(v) for v in row]
    if label in classes and len(classes) == len(values):
        return values[classes.index(label)]
    return max(values)


def _labels(model: object | None, texts: list[str]) -> list[str | None] | None:
    if model is None:
        return None
    return [None if label is None or str(label).strip().lower() in _NO_LABEL else str(label) for label in model.predict(texts)]


def _flags(model: object | None, texts: list[str]) -> list[bool] | None:
    if model is None:
        return None
    return [
        str(label).strip().lower() in {"1", "true", "yes"} if isinstance(label, str) else bool(label)
        for label in model.predict(texts)
    ]


def load(model_dir: Path) -> StubModel | SklearnModel:
    """Load model/model.joblib, or the stand-in when it is absent.

    A file that exists but fails to load raises: the service must not quietly
    serve the stand-in in place of the real model.
    """
    path = model_dir / "model.joblib"
    if not path.exists():
        logger.warning("model/model.joblib not found; serving the keyword stand-in")
        return StubModel()

    import joblib

    bundle = joblib.load(path)
    version = bundle.get("model_version") if isinstance(bundle, dict) else None
    if not version:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        version = f"sha256-{digest[:12]}"
    model = SklearnModel(bundle, str(version))
    model.predict([ticket_text("", "warm-up")])
    logger.info("loaded %s (version %s)", path, model.model_version)
    return model
