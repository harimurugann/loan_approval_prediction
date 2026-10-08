"""Analytics page: distributions, outcome relationships, and correlations."""

import pandas as pd
import streamlit as st

from components import charts, ui
from utils.analysis import binned_rate, rate_by_term
from utils.config import FEATURE_LABELS, TARGET
from utils.resources import load_dataset_or_stop, load_dataset_profile_or_stop


def _distributions(df: pd.DataFrame, profile: dict) -> None:
    counts = profile["class_counts"]
    first, second = st.columns(2)
    first.plotly_chart(charts.target_distribution(counts))
    term_counts = df["term"].value_counts().sort_index()
    second.plotly_chart(
        charts.counts_bar(
            [f"{int(term)} months" for term in term_counts.index],
            term_counts.tolist(),
            "Term distribution",
        )
    )
    third, fourth = st.columns(2)
    third.plotly_chart(charts.histogram(df, "loan_amnt"))
    fourth.plotly_chart(charts.histogram(df, "annual_inc"))
    fifth, sixth = st.columns(2)
    fifth.plotly_chart(charts.histogram(df, "int_rate"))
    sixth.plotly_chart(charts.histogram(df, "dti"))


def _relationships(df: pd.DataFrame, overall: float) -> None:
    ui.callout(
        "Reading these charts",
        " Each bar is the share of loans paid back within a group of similar loans. "
        "The axis runs from 0 to 100% so differences are not exaggerated.",
        kind="info",
    )
    first, second = st.columns(2)
    first.plotly_chart(
        charts.rate_bar(binned_rate(df, "loan_amnt", decimals=0), overall,
                        "Repayment rate by loan amount", FEATURE_LABELS["loan_amnt"])
    )
    second.plotly_chart(
        charts.rate_bar(binned_rate(df, "int_rate", decimals=1), overall,
                        "Repayment rate by interest rate", FEATURE_LABELS["int_rate"])
    )
    st.plotly_chart(
        charts.rate_bar(rate_by_term(df), overall, "Repayment rate by term", "Term")
    )


def _data_quality(profile: dict) -> None:
    missing = {k: v for k, v in profile["missing"].items() if v}
    first, second = st.columns(2)
    with first:
        ui.section_title("Missing values and duplicates")
        ui.render_card(
            ui.key_value_table(
                [(FEATURE_LABELS.get(column, column), f"{count:,}") for column, count in missing.items()]
                + [("Duplicate rows", f"{profile['duplicate_rows']:,}")]
            )
        )
        st.caption("Missing values are filled with the training median inside the model pipeline.")
    with second:
        ui.section_title("IQR outliers per feature")
        outliers = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS[column] for column in profile["iqr_outliers"]],
                "Outliers": list(profile["iqr_outliers"].values()),
            }
        )
        st.dataframe(outliers, hide_index=True)
    if profile["open_exceeds_total_rows"]:
        ui.callout(
            "Inconsistent account counts",
            f" {profile['open_exceeds_total_rows']:,} rows report more open accounts than "
            "total accounts, which is not possible for real credit files. The rows were kept "
            "for training, but the prediction form rejects this combination.",
            kind="warn",
        )


def render() -> None:
    """Render the analytics page."""
    df = load_dataset_or_stop()
    profile = load_dataset_profile_or_stop()
    overall = profile["positive_rate"]

    ui.page_header("Analytics", "Explore the historical loan data behind the model.")
    ui.render_metric_row(
        [
            ("Loans", f"{profile['rows']:,}", ""),
            ("Repayment rate", f"{overall:.1%}", ""),
            ("Median loan amount", f"{df['loan_amnt'].median():,.0f}", ""),
            ("Median annual income", f"{df['annual_inc'].median():,.0f}", ""),
        ]
    )
    st.write("")

    tabs = st.tabs(["Distributions", "Outcome relationships", "Correlations", "Data quality"])
    with tabs[0]:
        _distributions(df, profile)
    with tabs[1]:
        _relationships(df, overall)
    with tabs[2]:
        st.plotly_chart(charts.correlation_heatmap(df))
        strongest = (
            df.drop(columns=[TARGET]).corrwith(df[TARGET]).abs().sort_values(ascending=False)
        )
        ui.callout(
            "Feature-to-outcome correlation",
            f" The strongest correlation between any feature and the outcome is "
            f"{strongest.iloc[0]:.3f} ({FEATURE_LABELS[strongest.index[0]]}). Values this close "
            "to zero mean no feature is linearly related to repayment in this data.",
            kind="info",
        )
    with tabs[3]:
        _data_quality(profile)


render()
