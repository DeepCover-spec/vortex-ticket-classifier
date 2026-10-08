"""P3: train, evaluate and ship the improved ticket model.

Run from the repo root with the pinned environment (requirements.txt):

    python training/train_p3.py

What it does, in order:

1. Fits every head on train.csv and scores it on validation.csv. These are the
   numbers in the report and in training/experiments.md. Train-set CV is not used
   for selection: train repeats near-identical sentences, so CV sits near 0.99
   while validation, which is phrased differently, sits near 0.67.
2. Scores the shipped baseline (baseline-01) the same way, so the comparison is
   like for like.
3. Writes the evaluation charts to training/outputs/ and the dashboard figures
   to frontend/metrics.json.
4. Refits every head on train + validation (more phrasings for the hidden test
   set) and writes model/model.joblib and model/labels.json.

Model (P3-01):
- category label: word unigram + char_wb(2,5) TF-IDF, LinearSVC(C=0.5)
- confidence: the same features, LinearSVC wrapped in sigmoid calibration (5-fold).
  The label comes from the uncalibrated head because calibration's internal folds
  move the argmax and cost 1.5 to 3.5 macro-F1 points; the calibrated head is only
  asked "how sure is it of that label".
- secondary category: logistic regression, predicted only when P(none) < 0.65
- urgency: baseline-01's head, unchanged (it scored best)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline, make_union
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.predictor import REVIEW_THRESHOLD  # noqa: E402
from app.textproc import build_input, clean_text  # noqa: E402

DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
OUT = ROOT / "training" / "outputs"
SECONDARY_NONE_THRESHOLD = 0.65
warnings.filterwarnings("ignore")

CATEGORY_TO_TEAM = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}

# Chart colours: reference categorical slots 1-2, text inks, quiet grid.
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- features / heads
def word(ngrams=(1, 1)):
    return TfidfVectorizer(preprocessor=clean_text, analyzer="word", ngram_range=ngrams, min_df=2,
                           sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")


def chars():
    return TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=300_000)


def p3_heads() -> dict:
    """The P3 bundle, unfitted."""
    feats = lambda: make_union(word(), chars())  # noqa: E731
    return {
        "category_label": Pipeline([("features", feats()), ("clf", LinearSVC(C=0.5))]),
        "category": Pipeline([("features", feats()),
                              ("clf", CalibratedClassifierCV(LinearSVC(C=0.5), method="sigmoid", cv=5))]),
        "secondary": Pipeline([("features", feats()),
                               ("clf", LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))]),
        "urgent": Pipeline([("features", make_union(word((1, 2)), chars())),
                            ("clf", LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))]),
    }


def baseline_heads() -> dict:
    """baseline-01 as shipped by P2, for a like-for-like comparison."""
    feats = lambda: make_union(word((1, 2)), chars())  # noqa: E731
    return {
        "category": Pipeline([("features", feats()), ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"), method="sigmoid", cv=5))]),
        "secondary": Pipeline([("features", feats()),
                               ("clf", LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))]),
        "urgent": Pipeline([("features", feats()),
                            ("clf", LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))]),
    }


def fit(heads: dict, df: pd.DataFrame) -> dict:
    y = {"category_label": df.category, "category": df.category,
         "secondary": df.secondary_category.replace("", "none"), "urgent": df.is_urgent}
    return {name: model.fit(df.x, y[name]) for name, model in heads.items()}


# ---------------------------------------------------------------- prediction (mirrors app/predictor.py)
def predict(bundle: dict, texts, none_below: float | None):
    proba = bundle["category"].predict_proba(texts)
    classes = [str(c) for c in bundle["category"].classes_]
    if "category_label" in bundle:
        cats = np.array([str(c) for c in bundle["category_label"].predict(texts)])
    else:
        cats = np.array([classes[j] for j in proba.argmax(1)])
    conf = np.array([proba[i, classes.index(c)] for i, c in enumerate(cats)])
    sp = bundle["secondary"].predict_proba(texts)
    sc = [str(c) for c in bundle["secondary"].classes_]
    ni = sc.index("none")
    secs = []
    for i, cat in enumerate(cats):
        if none_below is None:
            top = sc[int(sp[i].argmax())]
            sec = None if top in ("none", cat) else top
        elif sp[i][ni] < none_below:
            ranked = np.argsort(-sp[i])
            sec = next((sc[j] for j in ranked if j != ni and sc[j] != cat), None)
        else:
            sec = None
        secs.append(sec)
    urgent = np.array([str(u).lower() in ("true", "1") for u in bundle["urgent"].predict(texts)])
    spam = cats == "spam_irrelevant"
    urgent[spam] = False
    secs = [None if s else v for s, v in zip(spam, secs)]
    return cats, conf, secs, urgent


def ece(conf, ok, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(((conf > lo) & (conf <= hi)).mean() * abs(ok[(conf > lo) & (conf <= hi)].mean()
                                                            - conf[(conf > lo) & (conf <= hi)].mean())
                     for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any()))


def score(name: str, bundle: dict, val: pd.DataFrame, none_below) -> dict:
    cats, conf, secs, urgent = predict(bundle, val.x, none_below)
    ok = cats == val.category.values
    sec_true = val.secondary_category.replace("", "none").values
    sec_pred = np.array([s or "none" for s in secs])
    has = sec_true != "none"
    p, r, f, _ = precision_recall_fscore_support(val.is_urgent, urgent, average="binary")
    m = {
        "name": name,
        "macro_f1": f1_score(val.category, cats, average="macro"),
        "accuracy": accuracy_score(val.category, cats),
        "secondary_exact_accuracy": float((sec_pred == sec_true).mean()),
        "secondary_recall": float((sec_pred[has] == sec_true[has]).mean()),
        "secondary_false_positive_rate": float((sec_pred[~has] != "none").mean()),
        "urgent_precision": p, "urgent_recall": r, "urgent_f1": f,
        "ece": ece(conf, ok), "confidence_auroc": roc_auc_score(ok, conf),
        "mean_confidence": float(conf.mean()),
    }
    print(f"{name:12s} macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} | secondary recall "
          f"{m['secondary_recall']:.3f} FP {m['secondary_false_positive_rate']:.3f} | urgent F1 {f:.3f} | "
          f"ECE {m['ece']:.3f} AUROC {m['confidence_auroc']:.3f}", flush=True)
    return {**m, "_pred": cats, "_conf": conf, "_ok": ok}


# ---------------------------------------------------------------- charts
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_progress(path: Path):
    """Validation macro-F1 per logged run, read from experiments.md (the single source)."""
    rows = []
    for line in (ROOT / "training" / "experiments.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and re.fullmatch(r"(baseline|exp|p3)-[\w-]+", cells[0]):
            try:
                rows.append((cells[0], float(cells[4])))
            except ValueError:
                pass
    if not rows:
        return
    ids, f1s = zip(*rows)
    base = dict(rows).get("baseline-01")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(ids) + 1.2))
    colors = [ORANGE if i.startswith("p3-01") else BLUE for i in ids]
    ax.barh(range(len(ids)), f1s, color=colors, height=0.6)
    ax.set_yticks(range(len(ids)), ids)
    ax.invert_yaxis()
    lo = min(f1s) - 0.03
    ax.set_xlim(lo, max(f1s) + 0.02)
    if base:
        ax.axvline(base, color=MUTED, linestyle="--", linewidth=1)
        ax.text(base, -0.9, " baseline-01", color=MUTED, fontsize=8, va="bottom")
    for y, v in enumerate(f1s):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Validation macro-F1 (fit on train)", color=MUTED)
    ax.set_title("Every logged run; shipped P3 model in orange", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_per_class(base: dict, new: dict, val: pd.DataFrame, path: Path):
    labels = sorted(CATEGORY_TO_TEAM)
    fb = f1_score(val.category, base["_pred"], labels=labels, average=None)
    fn = f1_score(val.category, new["_pred"], labels=labels, average=None)
    order = np.argsort(fn)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(y - 0.19, fb[order], height=0.36, color=BLUE, label="baseline-01")
    ax.barh(y + 0.19, fn[order], height=0.36, color=ORANGE, label="P3-01 (shipped)")
    ax.set_yticks(y, [labels[i] for i in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation F1 per category", color=MUTED)
    ax.set_title("Per-category F1: baseline vs P3", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_confusion(new: dict, val: pd.DataFrame, path: Path):
    labels = sorted(CATEGORY_TO_TEAM)
    cm = confusion_matrix(val.category, new["_pred"], labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("True", color=MUTED)
    ax.set_title("P3-01 confusion matrix on validation (row-normalised)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_reliability(new: dict, path: Path):
    conf, ok = new["_conf"], new["_ok"]
    edges = np.linspace(0, 1, 11)
    mids, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            mids.append(conf[m].mean()); accs.append(ok[m].mean()); ns.append(m.sum())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    ax.plot(mids, accs, marker="o", color=BLUE, linewidth=2, markersize=6)
    ax.set_xlabel("Stated confidence", color=MUTED)
    ax.set_ylabel("Actual accuracy", color=MUTED)
    ax.set_title(f"Calibration (ECE {new['ece']:.3f})", loc="left", fontsize=11, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _style(ax); ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax = axes[1]
    ths = np.linspace(0.2, 0.8, 25)
    ax.plot(ths, [ok[conf >= t].mean() for t in ths], color=BLUE, linewidth=2, label="accuracy of auto-routed")
    ax.plot(ths, [(conf >= t).mean() for t in ths], color=ORANGE, linewidth=2, label="share auto-routed")
    ax.axvline(REVIEW_THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
    ax.text(REVIEW_THRESHOLD, 0.95, f" human review below {REVIEW_THRESHOLD}", color=MUTED, fontsize=8)
    ax.set_xlabel("Confidence threshold", color=MUTED)
    ax.set_ylim(0, 1)
    ax.set_title("Human-review trade-off", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax); ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv", keep_default_na=False)
    val = pd.read_csv(DATA / "validation.csv", keep_default_na=False)
    for df in (train, val):
        df["x"] = [build_input(s, t) for s, t in zip(df.subject, df.text)]
        df["is_urgent"] = df.is_urgent.astype(str).str.lower().eq("true")
    print(f"scikit-learn {sklearn.__version__} | train {len(train)} | validation {len(val)}")

    print("\n1) Fit on train, score on validation")
    base = score("baseline-01", fit(baseline_heads(), train), val, None)
    new_bundle = fit(p3_heads(), train)
    new = score("P3-01", new_bundle, val, SECONDARY_NONE_THRESHOLD)

    print("\n2) Charts and dashboard metrics")
    chart_progress(OUT / "p3_progress.png")
    chart_per_class(base, new, val, OUT / "p3_per_class_f1.png")
    chart_confusion(new, val, OUT / "p3_confusion.png")
    chart_reliability(new, OUT / "p3_calibration.png")
    labels = sorted(CATEGORY_TO_TEAM)
    per_class = f1_score(val.category, new["_pred"], labels=labels, average=None)
    public = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
              for k, v in new.items() if not k.startswith("_") and k != "name"}
    dashboard = {
        **{k: public[k] for k in ("accuracy", "macro_f1")},
        "baseline_macro_f1": round(float(base["macro_f1"]), 4),
        "categories": labels,
        "per_class_f1": {c: round(float(f), 4) for c, f in zip(labels, per_class)},
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(val.category, new["_pred"], labels=labels).tolist(),
        "secondary_recall": public["secondary_recall"],
        "urgent_f1": public["urgent_f1"],
        "calibration_ece": public["ece"],
        "model_type": "TF-IDF (word unigrams + character 2-5 grams) with Linear SVM; sigmoid-calibrated confidence",
        "features": "Word unigrams and character n-grams (2-5), shared cleaner masks reference codes and digits",
        "training_approach": "Fit on train.csv, selected on validation.csv, shipped model refit on train + validation",
        "evaluation_metric": "Macro-F1 on validation.csv (fit on train only)",
    }
    (ROOT / "frontend").mkdir(exist_ok=True)
    (ROOT / "frontend" / "metrics.json").write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    (OUT / "p3_validation_metrics.json").write_text(json.dumps(
        {"baseline-01": {k: round(float(v), 4) for k, v in base.items() if not k.startswith("_") and k != "name"},
         "p3-01": {k: round(float(v), 4) for k, v in public.items() if isinstance(v, (int, float))}},
        indent=2), encoding="utf-8")

    print("\n3) Refit on train + validation and save the shipped model")
    full = pd.concat([train, val], ignore_index=True)
    final = fit(p3_heads(), full)
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "model.joblib"
    joblib.dump(final, path, compress=3)
    version = f"p3-tfidf-{hashlib.sha256(path.read_bytes()).hexdigest()[:8]}"
    (MODEL_DIR / "labels.json").write_text(json.dumps({
        "model_version": version,
        "sklearn_version": sklearn.__version__,
        "categories": [str(c) for c in final["category"].classes_],
        "category_to_team": CATEGORY_TO_TEAM,
        "refit_on_train_plus_val": True,
        "secondary_none_threshold": SECONDARY_NONE_THRESHOLD,
        "validation_scores_fit_on_train": {k: public[k] for k in ("macro_f1", "accuracy", "secondary_recall",
                                                                  "urgent_f1", "ece")},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    from app.predictor import TicketModel  # reload exactly as the API does
    check = TicketModel(MODEL_DIR).predict(list(val.text.head(20)), list(val.subject.head(20)))
    assert len(check) == 20 and all(r["model_version"] == version for r in check)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB) as {version}; reload OK | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
