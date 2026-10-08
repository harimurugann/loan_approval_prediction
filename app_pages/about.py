"""About page: methodology, limitations, and responsible use."""

import streamlit as st

from components import ui
from utils.config import FEATURE_DESCRIPTIONS, FEATURE_LABELS, FEATURES, TARGET
from utils.resources import load_model_or_stop


def render() -> None:
    """Render the about page."""
    _, report = load_model_or_stop()
    profile = report["data_profile"]

    ui.page_header("About this project", "What it does, how it was built, and where it falls short.")

    left, right = st.columns(2, gap="large")
    with left:
        ui.section_title("Overview and business problem")
        st.markdown(
            "Lenders want to know how likely a loan is to be repaid. This project trains a "
            "classifier on historical loan records and lets you score a new loan. "
            f"The target column, `{TARGET}`, records whether a loan was paid back, so the "
            "app predicts **repayment**, not a bank's real approval decision."
        )
        ui.section_title("Dataset")
        st.markdown(
            f"{profile['rows']:,} loans, {len(FEATURES)} features, and one target. "
            f"{profile['positive_rate']:.1%} of loans were paid back."
        )
        st.markdown("\n".join(
            f"- **{FEATURE_LABELS[feature]}** (`{feature}`): {FEATURE_DESCRIPTIONS[feature]}"
            for feature in FEATURES
        ))
    with right:
        ui.section_title("Methodology")
        st.markdown(
            f"1. Validate the CSV and confirm `{TARGET}` is binary.\n"
            f"2. Split into {1 - report['test_size']:.0%} training and {report['test_size']:.0%} "
            "test data, stratified by outcome, before any fitting.\n"
            "3. Build pipelines with median imputation (and scaling for logistic regression).\n"
            "4. Compare Logistic Regression, Random Forest, and Hist Gradient Boosting with "
            f"{report['cv_folds']}-fold cross-validation on the training data only.\n"
            "5. Select using ROC-AUC and the one-standard-error rule, preferring the simpler model.\n"
            "6. Evaluate once on the test set and save the full pipeline."
        )
        ui.section_title("Selected model")
        st.markdown(f"**{report['selected_model']}**. {report['selection_rationale']}")

    ui.section_title("Limitations")
    ui.callout(
        "Weak signal in the data" if report["signal_warning"] else "Interpret with care",
        f" Test ROC-AUC is {report['test_metrics']['roc_auc']:.3f}. Every feature has a "
        "near-zero correlation with the outcome, which suggests the data may be synthetic or "
        "that these features carry little information about repayment.",
        kind="warn",
    )
    st.markdown(
        f"- {profile['open_exceeds_total_rows']:,} rows have more open accounts than total "
        "accounts, and installments do not follow from loan amount, rate, and term, so the data "
        "is not internally consistent.\n"
        f"- Missing income ({profile['missing']['annual_inc']:,} rows) and DTI "
        f"({profile['missing']['dti']:,} rows) are filled with medians.\n"
        "- The model uses only eight fields and ignores credit history, employment, and economic conditions.\n"
        "- Predictions outside the training ranges are rejected rather than extrapolated."
    )

    ui.section_title("Responsible use")
    ui.callout(
        "Not a lending decision tool",
        " This project is for demonstration and portfolio purposes. Predictions come from "
        "historical patterns, are not guaranteed, and may reflect bias in the training data. "
        "Real lending decisions require financial, regulatory, and human review.",
        kind="info",
    )


render()
