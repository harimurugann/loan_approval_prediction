"""Prediction from validated user input using the saved pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
from sklearn.pipeline import Pipeline

from utils.config import DECISION_THRESHOLD, FEATURES


class PredictionError(RuntimeError):
    """Raised when the model cannot produce a prediction."""


@dataclass(frozen=True)
class PredictionResult:
    """Outcome of one prediction."""

    label: int
    probability_paid_back: float
    threshold: float


def predict_repayment(pipeline: Pipeline, values: Mapping[str, Any]) -> PredictionResult:
    """Run one applicant through the saved pipeline (same steps as training)."""
    try:
        frame = pd.DataFrame([{feature: float(values[feature]) for feature in FEATURES}])
        probability = float(pipeline.predict_proba(frame[FEATURES])[0, 1])
    except Exception as exc:  # noqa: BLE001 - surface one clean error to the UI
        raise PredictionError("The model could not score this input.") from exc
    return PredictionResult(
        label=int(probability >= DECISION_THRESHOLD),
        probability_paid_back=probability,
        threshold=DECISION_THRESHOLD,
    )
