"""Model construction, training, evaluation, and persistence."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import (
    CV_FOLDS,
    DECISION_THRESHOLD,
    FEATURES,
    MIN_USEFUL_AUC,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_PATH,
    TARGET,
    TEST_SIZE,
)
from utils.data_loader import load_dataset, profile_dataset

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Not paid back", "Paid back"]


class ModelError(RuntimeError):
    """Raised when saved model artifacts are missing, corrupt, or incompatible."""


def build_candidates() -> dict[str, Pipeline]:
    """Return candidate pipelines ordered from simplest to most complex.

    Every pipeline starts with a median imputer, so missing values seen at
    training time are handled identically at prediction time.
    """
    return {
        "Logistic Regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=25,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Hist Gradient Boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=150,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def select_model(cv_results: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Pick a model using the one-standard-error rule on cross-validated ROC-AUC.

    The simplest candidate whose mean AUC is within one standard deviation of
    the best score wins. This avoids choosing a complex model because of noise.
    """
    best_name = max(cv_results, key=lambda name: cv_results[name]["cv_auc_mean"])
    best = cv_results[best_name]
    cutoff = best["cv_auc_mean"] - best["cv_auc_std"]
    for name, scores in cv_results.items():
        if scores["cv_auc_mean"] >= cutoff:
            rationale = (
                f"Highest cross-validated ROC-AUC was {best['cv_auc_mean']:.3f} "
                f"({best_name}). '{name}' is the simplest candidate within one "
                f"standard deviation ({best['cv_auc_std']:.3f}) of that score, so it was chosen."
            )
            return name, rationale
    raise ModelError("No candidate model could be selected.")


def evaluate_model(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute all test-set metrics and curve data from real predictions."""
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "roc_curve": {
            "fpr": fpr[keep].round(5).tolist(),
            "tpr": tpr[keep].round(5).tolist(),
        },
    }


def compute_feature_importance(
    pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> list[dict[str, Any]]:
    """Permutation importance on held-out data (drop in ROC-AUC when shuffled)."""
    result = permutation_importance(
        pipeline,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    rows = [
        {"feature": feature, "importance": float(mean), "std": float(std)}
        for feature, mean, std in zip(FEATURES, result.importances_mean, result.importances_std)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def summarise_features(x_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-feature ranges from training data; used for form limits and validation."""
    summary: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        column = x_train[feature]
        summary[feature] = {
            "min": float(column.min()),
            "max": float(column.max()),
            "median": float(column.median()),
            "lower_bound": math.floor(column.min()),
            "upper_bound": math.ceil(column.max()),
        }
    return summary


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    """Train candidates, select one, and evaluate it once on the held-out test set.

    Leakage controls: the split happens before any fitting; imputation and
    scaling live inside each pipeline so they are fitted on training folds only;
    model selection uses cross-validation on the training set alone; the test
    set is touched exactly once, for the final metrics.
    """
    profile = profile_dataset(df)
    data = df.drop_duplicates().reset_index(drop=True)
    features, target = data[FEATURES], data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results: dict[str, dict[str, float]] = {}
    for name, candidate in build_candidates().items():
        scores = cross_val_score(candidate, x_train, y_train, cv=folds, scoring="roc_auc")
        cv_results[name] = {
            "cv_auc_mean": float(scores.mean()),
            "cv_auc_std": float(scores.std()),
        }
        logger.info("%s CV ROC-AUC: %.4f", name, scores.mean())

    selected, rationale = select_model(cv_results)
    pipeline = build_candidates()[selected]
    pipeline.fit(x_train, y_train)

    test_metrics = evaluate_model(pipeline, x_test, y_test)
    majority_class = int(y_train.mean() >= 0.5)
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "selected_model": selected,
        "selection_rationale": rationale,
        "cv_results": cv_results,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "decision_threshold": DECISION_THRESHOLD,
        "baseline_accuracy": float((y_test == majority_class).mean()),
        "test_metrics": test_metrics,
        "signal_warning": bool(test_metrics["roc_auc"] < MIN_USEFUL_AUC),
        "feature_importance": compute_feature_importance(pipeline, x_test, y_test),
        "feature_summary": summarise_features(x_train),
        "data_profile": profile,
    }
    return pipeline, report


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars to plain Python values when serialising."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def save_artifacts(pipeline: Pipeline, report: dict[str, Any]) -> None:
    """Write the fitted pipeline and training report using atomic replacement."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_tmp = MODEL_PATH.with_suffix(".tmp")
    report_tmp = REPORT_PATH.with_suffix(".tmp")
    joblib.dump(pipeline, model_tmp)
    report_tmp.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    os.replace(model_tmp, MODEL_PATH)
    os.replace(report_tmp, REPORT_PATH)


def load_artifacts() -> tuple[Pipeline, dict[str, Any]]:
    """Load the saved pipeline and report, raising ModelError if unusable.

    Only load model files that this project produced: joblib uses pickle.
    """
    if not MODEL_PATH.is_file() or not REPORT_PATH.is_file():
        raise ModelError("Saved model files were not found.")
    try:
        pipeline = joblib.load(MODEL_PATH)
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - unpickling can fail in many ways
        raise ModelError(f"Saved model files could not be read: {exc}") from exc

    if report.get("sklearn_version") != sklearn.__version__:
        raise ModelError("Saved model was built with a different scikit-learn version.")
    if not hasattr(pipeline, "predict_proba"):
        raise ModelError("Saved model file does not contain a valid classifier.")
    return pipeline, report


def ensure_artifacts() -> tuple[Pipeline, dict[str, Any]]:
    """Load the saved model, training and saving a new one if it is unusable."""
    try:
        return load_artifacts()
    except ModelError as exc:
        logger.info("Rebuilding model artifacts: %s", exc)
    pipeline, report = train_model(load_dataset())
    save_artifacts(pipeline, report)
    return pipeline, report
