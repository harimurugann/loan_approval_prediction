"""Loan prediction page: input form, validation, and result."""

import logging

import streamlit as st

from components import ui
from utils.config import FEATURE_DESCRIPTIONS, FEATURE_LABELS, VALID_TERMS
from utils.prediction import PredictionError, predict_repayment
from utils.resources import load_model_or_stop
from utils.validation import validate_inputs

logger = logging.getLogger(__name__)


def _float_input(feature: str, summary: dict, step: float, fmt: str = "%.2f") -> float:
    """Number input limited to the range seen in training, defaulting to the median."""
    stats = summary[feature]
    return st.number_input(
        FEATURE_LABELS[feature],
        min_value=float(stats["lower_bound"]),
        max_value=float(stats["upper_bound"]),
        value=float(round(stats["median"], 2)),
        step=step,
        format=fmt,
        help=FEATURE_DESCRIPTIONS[feature],
    )


def _int_input(feature: str, summary: dict) -> int:
    stats = summary[feature]
    return st.number_input(
        FEATURE_LABELS[feature],
        min_value=int(stats["lower_bound"]),
        max_value=int(stats["upper_bound"]),
        value=int(round(stats["median"])),
        step=1,
        help=FEATURE_DESCRIPTIONS[feature],
    )


def _render_form(summary: dict) -> tuple[bool, dict]:
    """Draw the form and return (submitted, values)."""
    with st.form("loan_form"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Loan details**")
            loan_amnt = _float_input("loan_amnt", summary, step=100.0)
            term = st.selectbox(
                FEATURE_LABELS["term"],
                options=list(VALID_TERMS),
                format_func=lambda months: f"{months} months",
                help=FEATURE_DESCRIPTIONS["term"],
            )
            int_rate = _float_input("int_rate", summary, step=0.1)
            installment = _float_input("installment", summary, step=10.0)
        with right:
            st.markdown("**Applicant profile**")
            annual_inc = _float_input("annual_inc", summary, step=1000.0)
            dti = _float_input("dti", summary, step=0.5)
            open_acc = _int_input("open_acc", summary)
            total_acc = _int_input("total_acc", summary)
        submitted = st.form_submit_button("Predict repayment", type="primary")

    values = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "annual_inc": annual_inc,
        "dti": dti,
        "open_acc": open_acc,
        "total_acc": total_acc,
    }
    return submitted, values


def _format_value(feature: str, value: float) -> str:
    if feature == "term":
        return f"{int(value)} months"
    if feature in ("open_acc", "total_acc"):
        return str(int(value))
    if feature in ("int_rate", "dti"):
        return f"{value:.2f}%"
    return f"{value:,.2f}"


def _render_result(pipeline, report: dict, values: dict) -> None:
    """Score the validated input and display the result card."""
    try:
        result = predict_repayment(pipeline, values)
    except PredictionError as exc:
        logger.exception("Prediction failed")
        st.error(f"{exc} Try again, or rebuild the model with `python train_model.py`.")
        return

    paid_back = result.label == 1
    ui.render_result_card(
        headline="Likely to be paid back" if paid_back else "Unlikely to be paid back",
        probability=result.probability_paid_back,
        kind="good" if paid_back else "bad",
        baseline_rate=report["data_profile"]["positive_rate"],
        threshold=result.threshold,
    )

    if report["signal_warning"]:
        ui.callout(
            "Low reliability",
            f" On held-out data this model scores ROC-AUC {report['test_metrics']['roc_auc']:.2f}, "
            "barely above chance. The estimate is close to the dataset's overall repayment "
            "rate and does not meaningfully distinguish this applicant from any other.",
            kind="warn",
        )

    ui.section_title("Input summary")
    ui.render_card(
        ui.key_value_table(
            (FEATURE_LABELS[feature], _format_value(feature, value))
            for feature, value in values.items()
        )
    )
    st.caption(
        f"Model: {report['selected_model']}, trained on {report['n_train']:,} loans. "
        "The probability is an estimate, not a guarantee of repayment or approval."
    )


def render() -> None:
    """Render the prediction page."""
    pipeline, report = load_model_or_stop()
    summary = report["feature_summary"]

    ui.page_header(
        "Loan Prediction",
        "Enter loan and applicant details to see the model's estimated chance that the "
        "loan is paid back.",
    )

    form_column, result_column = st.columns([3, 2], gap="large")
    with form_column:
        submitted, values = _render_form(summary)

    with result_column:
        if not submitted:
            ui.callout(
                "How to read the result",
                " The model outputs a probability that the loan is paid back, learned from "
                "historical records. It predicts the dataset's outcome, not a bank's approval decision.",
                kind="info",
            )
            return
        errors = validate_inputs(values, summary)
        if errors:
            for message in errors:
                st.error(message)
            return
        _render_result(pipeline, report, values)


render()
