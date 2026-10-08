"""Server-side validation of user-supplied loan inputs."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping

from utils.config import FEATURE_LABELS, FEATURES, VALID_TERMS

INTEGER_FEATURES = ("open_acc", "total_acc")
POSITIVE_FEATURES = ("loan_amnt", "int_rate", "installment", "annual_inc")


def _format_number(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def validate_inputs(
    values: Mapping[str, Any], feature_summary: Mapping[str, Mapping[str, float]]
) -> list[str]:
    """Return a list of human-readable problems; an empty list means valid.

    Range limits come from the training data, because predictions outside the
    range the model has seen would be extrapolation.
    """
    errors: list[str] = []
    for feature in FEATURES:
        label = FEATURE_LABELS[feature]
        value = values.get(feature)

        if not _is_number(value):
            errors.append(f"{label} is required and must be a number.")
            continue
        if not math.isfinite(value):
            errors.append(f"{label} must be a finite number.")
            continue
        if feature == "term":
            if value not in VALID_TERMS:
                terms = " or ".join(str(term) for term in VALID_TERMS)
                errors.append(f"{label} must be {terms} months.")
            continue
        if feature in POSITIVE_FEATURES and value <= 0:
            errors.append(f"{label} must be greater than zero.")
            continue
        if feature == "dti" and value < 0:
            errors.append(f"{label} cannot be negative.")
            continue
        if feature in INTEGER_FEATURES and (value < 0 or value != int(value)):
            errors.append(f"{label} must be a whole number of zero or more.")
            continue

        low = feature_summary[feature]["lower_bound"]
        high = feature_summary[feature]["upper_bound"]
        if not low <= value <= high:
            errors.append(
                f"{label} must be between {_format_number(low)} and {_format_number(high)} "
                "(the range the model was trained on)."
            )

    open_acc, total_acc = values.get("open_acc"), values.get("total_acc")
    if _is_number(open_acc) and _is_number(total_acc) and open_acc > total_acc:
        errors.append("Open accounts cannot exceed total accounts.")
    return errors
