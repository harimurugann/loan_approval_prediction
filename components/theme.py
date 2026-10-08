"""Global CSS for the application."""

import streamlit as st

CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.01em; }

.lp-header { margin-bottom: 1.4rem; }
.lp-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 .25rem; padding: 0; color: #12222E; }
.lp-header p { color: #5B6B7A; margin: 0; font-size: 1rem; max-width: 70ch; line-height: 1.5; }

.lp-section { margin: 1.6rem 0 .6rem; }
.lp-section h3 { font-size: 1.1rem; margin: 0; padding: 0; font-weight: 650; color: #12222E; }
.lp-section p { margin: .15rem 0 0; color: #5B6B7A; font-size: .9rem; }

.lp-card { background: #FFFFFF; border: 1px solid #E2E7EC; border-radius: 10px;
           padding: 1rem 1.15rem; box-shadow: 0 1px 2px rgba(16, 32, 46, .05); height: 100%; }
.lp-metric-label { color: #5B6B7A; font-size: .82rem; }
.lp-metric-value { font-size: 1.65rem; font-weight: 700; color: #12222E; line-height: 1.25; }
.lp-metric-note { color: #7A8794; font-size: .78rem; margin-top: .15rem; }

.lp-callout { border: 1px solid; border-left-width: 4px; border-radius: 10px;
              padding: .85rem 1rem; margin: .4rem 0 1rem; font-size: .92rem; line-height: 1.5; }
.lp-callout strong { display: block; margin-bottom: .15rem; }
.lp-callout.info { background: #EAF4F7; border-color: #B9D9E1; border-left-color: #0E6E7E; }
.lp-callout.warn { background: #FBF3E4; border-color: #EBD3A4; border-left-color: #B7791F; }
.lp-callout.good { background: #E9F5EF; border-color: #B7DCC8; border-left-color: #2E7D5B; }
.lp-callout.bad  { background: #FBECEB; border-color: #EBC1BE; border-left-color: #B4413B; }

.lp-result { background: #FFFFFF; border: 1px solid #E2E7EC; border-left: 6px solid #0E6E7E;
             border-radius: 10px; padding: 1.2rem 1.3rem; box-shadow: 0 1px 2px rgba(16, 32, 46, .05); }
.lp-result.good { border-left-color: #2E7D5B; }
.lp-result.bad { border-left-color: #B4413B; }
.lp-result .kicker { color: #5B6B7A; font-size: .82rem; }
.lp-result .headline { font-size: 1.5rem; font-weight: 700; margin: .1rem 0 .7rem; color: #12222E; }
.lp-result .prob { font-size: 2.1rem; font-weight: 700; color: #12222E; line-height: 1.1; }
.lp-result .prob-label { color: #5B6B7A; font-size: .85rem; margin-bottom: .5rem; }
.lp-bar { background: #E9EDF1; border-radius: 6px; height: 10px; overflow: hidden; }
.lp-bar > div { background: #0E6E7E; height: 100%; }
.lp-result .context { color: #5B6B7A; font-size: .85rem; margin-top: .6rem; line-height: 1.5; }

.lp-kv { width: 100%; border-collapse: collapse; font-size: .9rem; }
.lp-kv td { padding: .38rem 0; border-bottom: 1px solid #EDF0F3; }
.lp-kv td:last-child { text-align: right; font-weight: 600; color: #12222E; }
.lp-kv td:first-child { color: #5B6B7A; }
.lp-kv tr:last-child td { border-bottom: none; }

.lp-steps { margin: 0; padding-left: 1.2rem; color: #17212B; line-height: 1.7; font-size: .93rem; }

.stButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 8px; font-weight: 600; padding: .55rem 1.3rem;
}
[data-testid="stSidebar"] { border-right: 1px solid #E2E7EC; }
.lp-brand { font-weight: 700; font-size: 1.05rem; color: #12222E; margin: .2rem 0 0; }
.lp-brand-sub { color: #5B6B7A; font-size: .8rem; margin-bottom: .8rem; }

@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .lp-header h1 { font-size: 1.5rem; }
    .lp-metric-value { font-size: 1.3rem; }
}
</style>
"""


def inject_css() -> None:
    """Apply the global stylesheet once per page run."""
    st.markdown(CSS, unsafe_allow_html=True)
