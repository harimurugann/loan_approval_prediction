"""Home page: overview, key statistics, and entry point to predictions."""

import pandas as pd
import streamlit as st

from components import ui
from utils.config import FEATURE_DESCRIPTIONS, FEATURE_LABELS, FEATURES
from utils.resources import load_model_or_stop


def _feature_table(summary: dict) -> pd.DataFrame:
    rows = [
        {
            "Feature": FEATURE_LABELS[feature],
            "Description": FEATURE_DESCRIPTIONS[feature],
            "Min": summary[feature]["min"],
            "Median": summary[feature]["median"],
            "Max": summary[feature]["max"],
        }
        for feature in FEATURES
    ]
    return pd.DataFrame(rows)


def render() -> None:
    """Render the home page."""
    _, report = load_model_or_stop()
    profile = report["data_profile"]
    metrics = report["test_metrics"]

    ui.page_header(
        "Loan Repayment Prediction",
        "Estimate whether a loan is likely to be paid back, based on patterns in "
        "20,000 historical loan records.",
    )

    if report["signal_warning"]:
        ui.callout(
            "Read this first: the model has almost no predictive power on this dataset.",
            f" Its test ROC-AUC is {metrics['roc_auc']:.2f} (0.50 is a coin flip) and its "
            f"accuracy ({metrics['accuracy']:.1%}) equals simply guessing \"paid back\" every "
            f"time ({report['baseline_accuracy']:.1%}). Treat every estimate as illustrative.",
            kind="warn",
        )

    ui.render_metric_row(
        [
            ("Loan records", f"{profile['rows']:,}", "After validation"),
            ("Historical repayment rate", f"{profile['positive_rate']:.1%}", "Share with loan_paid_back = 1"),
            ("Input features", str(len(FEATURES)), "Loan and applicant details"),
            ("Test ROC-AUC", f"{metrics['roc_auc']:.3f}", f"{report['selected_model']}, held-out data"),
        ]
    )

    st.write("")
    if st.button("Make a prediction", type="primary", icon=":material/calculate:"):
        st.switch_page("app_pages/prediction.py")

    left, right = st.columns(2, gap="large")
    with left:
        ui.section_title("How it works")
        st.markdown(
            '<div class="lp-card"><ol class="lp-steps">'
            "<li>You enter the eight loan and applicant details.</li>"
            "<li>The inputs are checked against the ranges seen in the training data.</li>"
            "<li>A saved scikit-learn pipeline applies the same imputation and scaling used in training.</li>"
            "<li>The model returns an estimated probability that the loan is paid back.</li>"
            "</ol></div>",
            unsafe_allow_html=True,
        )
    with right:
        ui.section_title("Model overview")
        ui.render_card(
            ui.key_value_table(
                [
                    ("Selected model", report["selected_model"]),
                    ("Training rows", f"{report['n_train']:,}"),
                    ("Test rows", f"{report['n_test']:,}"),
                    ("Model selection", f"{report['cv_folds']}-fold cross-validation"),
                    ("Decision threshold", f"{report['decision_threshold']:.0%}"),
                ]
            )
        )

    ui.section_title("Dataset features", "Values are taken from the training split.")
    st.dataframe(
        _feature_table(report["feature_summary"]),
        hide_index=True,
        column_config={
            "Min": st.column_config.NumberColumn(format="%.2f"),
            "Median": st.column_config.NumberColumn(format="%.2f"),
            "Max": st.column_config.NumberColumn(format="%.2f"),
        },
    )


render()
