"""Load the trained ticket classifier.

The agreed predictor contract (team plan) is:

    predict(texts: list[str]) -> list[dict]
    each dict: {"category": str, "confidence": float}

Team is read from the fixed category-to-team table when that file exists.
This module never invents a category, a team, or a confidence score.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.joblib"
TEAM_MAP_PATH = ROOT / "model" / "category_teams.json"
LABELS_PATH = ROOT / "model" / "labels.json"


class ClassifierError(Exception):
    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def classifier_status() -> dict:
    """Report whether a real classifier can be called. Does not invent scores."""
    if _predictor_module() is not None:
        return {
            "classifier": "ready",
            "source": "predictor",
            "detail": "Using app.predictor.predict.",
        }
    if MODEL_PATH.is_file():
        return {
            "classifier": "ready",
            "source": "model",
            "detail": "Using model/model.joblib.",
        }
    return {
        "classifier": "unavailable",
        "source": None,
        "detail": (
            "The trained classifier is not loaded. Add app/predictor.py "
            "or model/model.joblib."
        ),
    }


def classify_texts(texts: list[str]) -> list[dict]:
    raw = _call_model(texts)
    if not isinstance(raw, list) or len(raw) != len(texts):
        raise ClassifierError(
            "The classifier returned a response the demo could not read.",
            502,
        )

    teams = _load_team_map()
    results: list[dict] = []
    for text, item in zip(texts, raw):
        results.append(_normalize_item(text, item, teams))
    return results


def _call_model(texts: list[str]) -> list:
    module = _predictor_module()
    if module is not None:
        predict = getattr(module, "predict", None)
        if not callable(predict):
            raise ClassifierError(
                "app/predictor.py must define predict(texts) -> list[dict].",
                502,
            )
        try:
            return predict(texts)
        except ClassifierError:
            raise
        except Exception:
            raise ClassifierError(
                "The trained classifier failed while scoring these tickets.",
                502,
            ) from None

    if MODEL_PATH.is_file():
        return _predict_joblib(texts)

    raise ClassifierError(
        "The trained classifier is not loaded yet. "
        "Classification will work once the model file is added.",
        503,
    )


def _predictor_module():
    try:
        return importlib.import_module("app.predictor")
    except ModuleNotFoundError as exc:
        if exc.name == "app.predictor":
            return None
        raise ClassifierError(
            "The classifier dependency could not be loaded.",
            503,
        ) from None
    except Exception:
        raise ClassifierError(
            "app/predictor.py could not be imported.",
            502,
        ) from None


def _predict_joblib(texts: list[str]) -> list[dict]:
    try:
        import joblib
    except ImportError:
        raise ClassifierError(
            "model/model.joblib is present, but joblib is not installed.",
            503,
        ) from None

    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception:
        raise ClassifierError(
            "The model file could not be loaded.",
            502,
        ) from None

    try:
        predicted = pipeline.predict(texts)
    except Exception:
        raise ClassifierError(
            "The trained classifier failed while scoring these tickets.",
            502,
        ) from None

    classes = _estimator_classes(pipeline)
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(texts)
        except Exception:
            probabilities = None

    results = []
    for index, category in enumerate(predicted):
        confidence = None
        if probabilities is not None and classes:
            row = probabilities[index]
            label = str(category)
            if label in classes:
                confidence = float(row[classes.index(label)])
            else:
                confidence = float(max(row))
        results.append({"category": str(category), "confidence": confidence})
    return results


def _estimator_classes(pipeline) -> list[str] | None:
    if hasattr(pipeline, "classes_"):
        return [str(label) for label in pipeline.classes_]
    steps = getattr(pipeline, "steps", None)
    if steps:
        final = steps[-1][1]
        if hasattr(final, "classes_"):
            return [str(label) for label in final.classes_]
    labels = _load_json(LABELS_PATH)
    if isinstance(labels, list) and all(isinstance(item, str) for item in labels):
        return labels
    if isinstance(labels, dict) and isinstance(labels.get("labels"), list):
        return [str(item) for item in labels["labels"]]
    return None


def _normalize_item(text: str, item, teams: dict[str, str]) -> dict:
    if not isinstance(item, dict) or "category" not in item:
        raise ClassifierError(
            "The classifier did not return a category.",
            502,
        )
    category = str(item["category"])
    team = item.get("team")
    if team is None and category in teams:
        team = teams[category]
    return {
        "ticket": text,
        "category": category,
        "team": None if team is None else str(team),
        "confidence": _clean_confidence(item.get("confidence")),
    }


def _clean_confidence(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _load_team_map() -> dict[str, str]:
    direct = _load_json(TEAM_MAP_PATH)
    mapping = _as_team_map(direct)
    if mapping:
        return mapping
    labels = _load_json(LABELS_PATH)
    if isinstance(labels, dict):
        for key in ("category_to_team", "teams", "team_map"):
            mapping = _as_team_map(labels.get(key))
            if mapping:
                return mapping
    return {}


def _as_team_map(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    mapping: dict[str, str] = {}
    for category, team in value.items():
        if isinstance(category, str) and isinstance(team, str):
            mapping[category] = team
    return mapping


def _load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
