"""Loan Repayment Prediction: Streamlit entry point.

Run from the project root with:  streamlit run app.py
"""

import logging

import streamlit as st

from components.theme import inject_css
from components.ui import DISCLAIMER, render_sidebar_status
from utils.resources import get_artifacts

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Loan Repayment Predictor",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _model_status() -> tuple[str | None, float | None]:
    """Return (model name, test ROC-AUC) for the sidebar, or Nones if unavailable."""
    try:
        _, report = get_artifacts()
    except Exception:  # noqa: BLE001 - pages show the detailed error
        logger.warning("Model status unavailable for sidebar", exc_info=True)
        return None, None
    return report["selected_model"], report["test_metrics"]["roc_auc"]


def main() -> None:
    """Configure navigation and run the selected page."""
    inject_css()
    navigation = st.navigation(
        [
            st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
            st.Page("app_pages/prediction.py", title="Loan Prediction",
                    icon=":material/calculate:", url_path="prediction"),
            st.Page("app_pages/analytics.py", title="Analytics",
                    icon=":material/bar_chart:", url_path="analytics"),
            st.Page("app_pages/model_performance.py", title="Model Performance",
                    icon=":material/monitoring:", url_path="model-performance"),
            st.Page("app_pages/about.py", title="About",
                    icon=":material/info:", url_path="about"),
        ]
    )
    render_sidebar_status(*_model_status())
    try:
        navigation.run()
    except Exception:  # noqa: BLE001 - last-resort guard against raw tracebacks
        logger.exception("Unhandled error while rendering a page")
        st.error("Something unexpected went wrong. Reload the page, or run "
                 "`python train_model.py` to rebuild the model files.")
    st.caption(DISCLAIMER)


main()
