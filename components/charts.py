"""Plotly figure builders with a consistent visual style."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go

from utils.analysis import RateSeries
from utils.config import FEATURE_LABELS, FEATURES, TARGET

PRIMARY = "#0E6E7E"
ACCENT = "#B7791F"
MUTED = "#9AA8B4"
FONT = "Segoe UI, system-ui, -apple-system, sans-serif"


def style_figure(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    """Apply shared layout settings."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 16, b=10),
        title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        font=dict(family=FONT, size=12, color="#17212B"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E9EDF1")
    return fig


def target_distribution(counts: dict[str, int]) -> go.Figure:
    """Bar chart of the two outcome classes."""
    total = sum(counts.values())
    values = [counts["1"], counts["0"]]
    fig = go.Figure(
        go.Bar(
            x=["Paid back", "Not paid back"],
            y=values,
            marker_color=[PRIMARY, ACCENT],
            text=[f"{value:,} ({value / total:.1%})" for value in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Loans", range=[0, max(values) * 1.15])
    return style_figure(fig, "Outcome distribution")


def histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """Histogram of one numeric column."""
    fig = go.Figure(
        go.Histogram(x=df[column].dropna(), nbinsx=40, marker_color=PRIMARY)
    )
    fig.update_layout(bargap=0.05)
    fig.update_xaxes(title=FEATURE_LABELS[column])
    fig.update_yaxes(title="Loans")
    return style_figure(fig, f"{FEATURE_LABELS[column]} distribution")


def counts_bar(labels: Sequence[str], counts: Sequence[int], title: str) -> go.Figure:
    """Simple bar chart of counts per category."""
    fig = go.Figure(go.Bar(x=list(labels), y=list(counts), marker_color=PRIMARY))
    fig.update_yaxes(title="Loans")
    return style_figure(fig, title)


def rate_bar(series: RateSeries, overall: float, title: str, x_title: str) -> go.Figure:
    """Repayment rate per group, on a full 0-100% axis to keep differences honest."""
    labels, rates, counts = series
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=rates,
            marker_color=PRIMARY,
            customdata=counts,
            hovertemplate="%{x}<br>Repayment rate: %{y:.1%}<br>Loans: %{customdata:,}<extra></extra>",
        )
    )
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color=ACCENT,
        annotation_text=f"Overall {overall:.1%}",
        annotation_position="bottom right",
    )
    fig.update_yaxes(title="Repayment rate", range=[0, 1], tickformat=".0%")
    fig.update_xaxes(title=x_title, tickangle=-35)
    return style_figure(fig, title, height=360)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Pearson correlation matrix of features and target."""
    columns = FEATURES + [TARGET]
    corr = df[columns].corr()
    labels = [FEATURE_LABELS.get(column, "Loan paid back") for column in columns]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            texttemplate="%{z:.2f}",
            hovertemplate="%{x} / %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-35)
    return style_figure(fig, "Correlation matrix", height=520)


def confusion_matrix_chart(matrix: dict[str, int]) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    z = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    labels = ["Not paid back", "Paid back"]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=[[0, "#F4F6F8"], [1, PRIMARY]],
            texttemplate="%{z:,}",
            textfont=dict(size=16),
            showscale=False,
            hovertemplate="Actual %{y}<br>Predicted %{x}: %{z:,}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return style_figure(fig, "Confusion matrix (test set)", height=340)


def roc_curve_chart(fpr: Sequence[float], tpr: Sequence[float], auc: float) -> go.Figure:
    """ROC curve with the chance diagonal."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=list(fpr), y=list(tpr), mode="lines", line=dict(color=PRIMARY, width=3),
                   name=f"Model (AUC {auc:.3f})")
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                   line=dict(color=MUTED, dash="dash"), name="Chance (AUC 0.500)")
    )
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    style_figure(fig, "ROC curve (test set)", height=340)
    fig.update_layout(showlegend=True, legend=dict(x=0.4, y=0.08))
    return fig


def importance_chart(rows: list[dict]) -> go.Figure:
    """Horizontal bar chart of permutation importance with error bars."""
    ordered = sorted(rows, key=lambda row: row["importance"])
    fig = go.Figure(
        go.Bar(
            x=[row["importance"] for row in ordered],
            y=[FEATURE_LABELS[row["feature"]] for row in ordered],
            orientation="h",
            marker_color=PRIMARY,
            error_x=dict(type="data", array=[row["std"] for row in ordered], color=MUTED),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Drop in ROC-AUC when the feature is shuffled")
    fig.update_yaxes(showgrid=False)
    return style_figure(fig, "Permutation feature importance (test set)", height=380)
