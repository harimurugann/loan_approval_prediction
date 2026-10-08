"""Model performance page: metrics computed on the held-out test set."""

import pandas as pd
import streamlit as st

from components import charts, ui
from utils.resources import load_model_or_stop

REPORT_ROWS = ["Not paid back", "Paid back", "macro avg", "weighted avg"]


def _classification_table(report_dict: dict) -> pd.DataFrame:
    table = pd.DataFrame(report_dict).T.loc[REPORT_ROWS]
    table = table.rename(columns={"f1-score": "F1-score", "precision": "Precision",
                                  "recall": "Recall", "support": "Support"})
    table["Support"] = table["Support"].astype(int)
    return table.round(3)


def _candidate_table(cv_results: dict, selected: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": name,
                "CV ROC-AUC (mean)": round(scores["cv_auc_mean"], 3),
                "CV ROC-AUC (std)": round(scores["cv_auc_std"], 3),
                "Selected": "Yes" if name == selected else "",
            }
            for name, scores in cv_results.items()
        ]
    )


def render() -> None:
    """Render the model performance page."""
    _, report = load_model_or_stop()
    metrics = report["test_metrics"]

    ui.page_header(
        "Model Performance",
        f"{report['selected_model']}, evaluated on {report['n_test']:,} loans that were "
        "never used for training or model selection.",
    )

    if report["signal_warning"]:
        ui.callout(
            "The model performs no better than chance.",
            f" ROC-AUC is {metrics['roc_auc']:.3f}, where 0.5 means random ranking. The "
            f"accuracy of {metrics['accuracy']:.1%} matches a baseline that always predicts "
            f"\"paid back\" ({report['baseline_accuracy']:.1%}), and the model never flags a "
            "loan as unlikely to be repaid. The eight features carry little information about "
            "the outcome in this dataset.",
            kind="warn",
        )

    ui.render_metric_row(
        [
            ("Accuracy", f"{metrics['accuracy']:.3f}", f"Baseline {report['baseline_accuracy']:.3f}"),
            ("Precision", f"{metrics['precision']:.3f}", "Of predicted paid back"),
            ("Recall", f"{metrics['recall']:.3f}", "Of actual paid back"),
            ("F1-score", f"{metrics['f1']:.3f}", "Precision and recall"),
            ("ROC-AUC", f"{metrics['roc_auc']:.3f}", "Chance level 0.500"),
        ]
    )
    st.write("")

    first, second = st.columns(2)
    first.plotly_chart(charts.confusion_matrix_chart(metrics["confusion_matrix"]))
    second.plotly_chart(
        charts.roc_curve_chart(
            metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], metrics["roc_auc"]
        )
    )

    st.plotly_chart(charts.importance_chart(report["feature_importance"]))
    st.caption("Importance values near zero mean shuffling that feature does not change ranking quality.")

    left, right = st.columns(2, gap="large")
    with left:
        ui.section_title("Classification report", "Per-class results on the test set.")
        st.dataframe(_classification_table(metrics["classification_report"]))
    with right:
        ui.section_title("Model comparison", f"{report['cv_folds']}-fold cross-validation on training data.")
        st.dataframe(_candidate_table(report["cv_results"], report["selected_model"]), hide_index=True)
        st.caption(report["selection_rationale"])

    with st.expander("Why these metrics matter for lending"):
        st.markdown(
            "- **Accuracy** can mislead when classes are imbalanced. Here about 80% of loans are "
            "paid back, so predicting \"paid back\" for everyone already scores about 80%.\n"
            "- **Precision** answers: of the loans predicted as paid back, how many were?\n"
            "- **Recall** answers: of the loans that were paid back, how many did the model find?\n"
            "- **F1-score** balances precision and recall, but on this data it is inflated by the "
            "majority class.\n"
            "- **ROC-AUC** measures how well the model ranks repaid loans above unpaid ones, "
            "independent of any threshold. It is the main metric used for model selection.\n"
            "- **Confusion matrix** shows the costly mistakes directly: unpaid loans predicted as "
            "paid back (false positives) and repaid loans wrongly flagged (false negatives)."
        )

    st.caption(
        f"Trained {report['trained_at']} with scikit-learn {report['sklearn_version']}. "
        f"Stratified {1 - report['test_size']:.0%}/{report['test_size']:.0%} split, "
        f"random state {report['random_state']}, decision threshold {report['decision_threshold']:.0%}."
    )


render()
