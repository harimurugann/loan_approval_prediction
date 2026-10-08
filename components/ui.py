"""Reusable presentational components."""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import streamlit as st

DISCLAIMER = (
    "Demonstration project. Predictions reflect historical patterns in one dataset "
    "and are not financial advice or a lending decision."
)


def page_header(title: str, subtitle: str) -> None:
    """Render the page title block."""
    st.markdown(
        f'<div class="lp-header"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def section_title(title: str, caption: str = "") -> None:
    """Render a section heading with an optional one-line caption."""
    caption_html = f"<p>{html.escape(caption)}</p>" if caption else ""
    st.markdown(
        f'<div class="lp-section"><h3>{html.escape(title)}</h3>{caption_html}</div>',
        unsafe_allow_html=True,
    )


def render_metric_row(items: Sequence[tuple[str, str, str]]) -> None:
    """Render (label, value, note) tuples as equal-width metric cards."""
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        note_html = f'<div class="lp-metric-note">{html.escape(note)}</div>' if note else ""
        column.markdown(
            f'<div class="lp-card"><div class="lp-metric-label">{html.escape(label)}</div>'
            f'<div class="lp-metric-value">{html.escape(value)}</div>{note_html}</div>',
            unsafe_allow_html=True,
        )


def callout(title: str, body: str, kind: str = "info") -> None:
    """Render a message box. ``kind`` is one of info, warn, good, bad."""
    st.markdown(
        f'<div class="lp-callout {kind}"><strong>{html.escape(title)}</strong>'
        f"{html.escape(body)}</div>",
        unsafe_allow_html=True,
    )


def key_value_table(rows: Iterable[tuple[str, str]]) -> str:
    """Return an HTML two-column table for label/value pairs."""
    body = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f'<table class="lp-kv">{body}</table>'


def render_card(inner_html: str) -> None:
    """Wrap pre-escaped HTML in a card."""
    st.markdown(f'<div class="lp-card">{inner_html}</div>', unsafe_allow_html=True)


def render_result_card(
    headline: str, probability: float, kind: str, baseline_rate: float, threshold: float
) -> None:
    """Show the prediction outcome and the model-estimated probability."""
    st.markdown(
        f'<div class="lp-result {kind}"><div class="kicker">Loan repayment prediction</div>'
        f'<div class="headline">{html.escape(headline)}</div>'
        f'<div class="prob">{probability:.1%}</div>'
        '<div class="prob-label">Model-estimated probability of repayment</div>'
        f'<div class="lp-bar"><div style="width:{probability * 100:.1f}%"></div></div>'
        f'<div class="context">Repayment rate across the whole dataset: {baseline_rate:.1%}. '
        f"The outcome is \"paid back\" when the estimate is at least {threshold:.0%}.</div></div>",
        unsafe_allow_html=True,
    )


def render_sidebar_status(model_name: str | None, roc_auc: float | None) -> None:
    """Show the brand block and model status in the sidebar."""
    with st.sidebar:
        st.markdown(
            '<div class="lp-brand">Loan Repayment Predictor</div>'
            '<div class="lp-brand-sub">Machine learning demonstration</div>',
            unsafe_allow_html=True,
        )
        if model_name is not None and roc_auc is not None:
            st.markdown(
                key_value_table([("Model", model_name), ("Test ROC-AUC", f"{roc_auc:.3f}")]),
                unsafe_allow_html=True,
            )
        st.caption(DISCLAIMER)
