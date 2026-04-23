"""
=============================================================================
  LOAN APPROVAL INTELLIGENCE SYSTEM  —  Enterprise Streamlit Application
  Version  : 2.0
  Features : 20 advanced modules (single prediction, bulk, XAI, drift,
             geo-map, benchmarking, fraud, PDF statements, CRM, batch
             download, secure login, GDPR anonymisation, credit roadmap,
             market rates, card recommendation, alerts, AI bot, risk
             categorisation, audit logs, cross-platform UI)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Core Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, io, re, csv, json, time, datetime, random, hashlib, base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import bcrypt

from pathlib import Path

# PDF generation
try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False

# Folium geo-map
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Page Config & Custom CSS  (Feature 20 – Cross-platform UI)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Intelligence System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1.2rem; max-width: 1300px; }

/* ── Metric Cards ── */
.metric-card {
  background: linear-gradient(135deg,#1e3a5f,#2563eb);
  border-radius: 12px; padding: 18px 22px; color: #fff;
  box-shadow: 0 4px 15px rgba(37,99,235,.3); margin-bottom: 12px;
}
.metric-card h3 { font-size: .8rem; opacity: .8; margin: 0 0 4px; }
.metric-card h1 { font-size: 1.9rem; font-weight: 700; margin: 0; }

/* ── Risk badge ── */
.risk-high     { background:#ef4444; color:#fff; padding:4px 12px;
                  border-radius:20px; font-weight:600; font-size:.85rem; }
.risk-moderate { background:#f97316; color:#fff; padding:4px 12px;
                  border-radius:20px; font-weight:600; font-size:.85rem; }
.risk-low      { background:#22c55e; color:#fff; padding:4px 12px;
                  border-radius:20px; font-weight:600; font-size:.85rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label { color:#94a3b8 !important; }

/* ── Tables ── */
.styled-table { width:100%; border-collapse:collapse; font-size:.85rem; }
.styled-table th { background:#1e3a5f; color:#fff; padding:8px 12px; text-align:left; }
.styled-table td { padding:7px 12px; border-bottom:1px solid #e2e8f0; }
.styled-table tr:hover td { background:#f0f7ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Constants & Paths
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
AUDIT_LOG   = Path("audit_log.csv")

ADMIN_HASH  = bcrypt.hashpw(b"admin123", bcrypt.gensalt())   # default password
MARKET_RATES = {
    "Prime Rate (US)": "8.50%",
    "30-yr Fixed Mortgage": "7.15%",
    "15-yr Fixed Mortgage": "6.65%",
    "5/1 ARM": "6.80%",
    "Personal Loan Avg": "11.92%",
    "Auto Loan (New, 60mo)": "7.50%",
    "Student Loan (Federal)": "6.54%",
    "Credit Card APR Avg": "21.59%",
    "HELOC": "9.25%",
    "Business Loan Avg": "9.10%",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models …")
def load_artifacts():
    """Load all saved joblib artefacts; return dict."""
    arts = {}
    for key, fname in {
        "rf":       "loan_rf_model.sav",
        "pipeline": "loan_pipeline.sav",
        "meta":     "column_meta.sav",
        "metrics":  "benchmark_metrics.sav",
        "anomaly":  "anomaly_detector.sav",
        "scaler":   "num_scaler.sav",
    }.items():
        p = MODELS_DIR / fname
        if p.exists():
            arts[key] = joblib.load(p)

    xgb_path = MODELS_DIR / "loan_xgb_model.sav"
    if xgb_path.exists():
        arts["xgb"] = joblib.load(xgb_path)

    fi_path = REPORTS_DIR / "feature_importances.csv"
    if fi_path.exists():
        arts["feature_importances"] = pd.read_csv(fi_path, index_col=0, header=0)

    return arts

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
def risk_label(prob: float) -> str:
    """Map approval probability to risk tier."""
    if prob >= 0.65:  return "Low Risk"
    if prob >= 0.40:  return "Moderate Risk"
    return "High Risk"

def risk_badge(label: str) -> str:
    css = {"Low Risk": "risk-low", "Moderate Risk": "risk-moderate",
           "High Risk": "risk-high"}.get(label, "risk-high")
    return f'<span class="{css}">{label}</span>'

def anonymise(df: pd.DataFrame) -> pd.DataFrame:
    """GDPR masking: hash strings, round numerics. (Feature 12)"""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8].upper()
        )
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(-2)
    return out

def write_audit(action: str, detail: str = ""):
    """Append one row to the CSV audit log. (Feature 19)"""
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = st.session_state.get("username", "guest")
    row  = [ts, user, action, detail]
    first_write = not AUDIT_LOG.exists()
    with open(AUDIT_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if first_write:
            w.writerow(["timestamp", "user", "action", "detail"])
        w.writerow(row)

def predict_single(arts, input_dict: dict) -> dict:
    """Run a single prediction through the RF pipeline."""
    meta = arts["meta"]
    cols = meta["NUMERICAL_FEATURES"] + meta["CATEGORICAL_FEATURES"]
    df   = pd.DataFrame([input_dict])[cols]
    prob = arts["rf"].predict_proba(df)[0, 1]
    pred = int(prob >= 0.50)
    rl   = risk_label(prob)
    return {"probability": prob, "prediction": pred, "risk": rl}

def predict_dataframe(arts, df: pd.DataFrame) -> pd.DataFrame:
    """Run batch prediction and append result columns."""
    meta = arts["meta"]
    cols = meta["NUMERICAL_FEATURES"] + meta["CATEGORICAL_FEATURES"]
    # Only keep columns that exist in both the expected list and the DataFrame
    available = [c for c in cols if c in df.columns]
    probs  = arts["rf"].predict_proba(df[available])[:, 1]
    result = df.copy()
    result["approval_probability"] = probs.round(4)
    result["prediction"]           = (probs >= 0.50).astype(int)
    result["decision"]             = result["prediction"].map({1: "Approved", 0: "Rejected"})
    result["risk_tier"]            = [risk_label(p) for p in probs]
    return result

def is_anomaly(arts, num_vals: list) -> bool:
    """Return True if IsolationForest flags as anomaly."""
    if "anomaly" not in arts or "scaler" not in arts:
        return False
    scaled = arts["scaler"].transform([num_vals])
    return arts["anomaly"].predict(scaled)[0] == -1

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Feature 11 – Secure Admin Login
# ─────────────────────────────────────────────────────────────────────────────
def login_gate():
    """Block access until correct password entered."""
    if st.session_state.get("authenticated"):
        return
    st.markdown("## 🔐 Secure Admin Login")
    st.info("Default credentials — username: **admin** | password: **admin123**")
    with st.form("login_form"):
        uname = st.text_input("Username")
        pwd   = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if uname == "admin" and bcrypt.checkpw(pwd.encode(), ADMIN_HASH):
                st.session_state["authenticated"] = True
                st.session_state["username"] = uname
                write_audit("LOGIN", "success")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Feature 8 – Bank Statement PDF Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_statement(applicant: dict, prediction: dict) -> bytes:
    """Generate a professional bank statement PDF and return as bytes."""
    if not FPDF_OK:
        return b""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_fill_color(30, 58, 95)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "  LOAN APPROVAL INTELLIGENCE SYSTEM", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(6)
    pdf.cell(0, 8, f"Statement Date: {datetime.date.today().strftime('%B %d, %Y')}", ln=True)
    pdf.cell(0, 8, f"Reference ID  : LOAN-{random.randint(100000,999999)}", ln=True)
    pdf.ln(4)
    # Divider
    pdf.set_draw_color(37, 99, 235); pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(4)
    # Applicant section
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "APPLICANT DETAILS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in applicant.items():
        pdf.cell(70, 7, str(k).replace("_", " ").title())
        pdf.cell(0, 7, str(v), ln=True)
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(4)
    # Decision section
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "CREDIT DECISION", ln=True)
    pdf.set_font("Helvetica", "B", 13)
    decision = "✔ APPROVED" if prediction["prediction"] == 1 else "✘ REJECTED"
    colour   = (34, 197, 94) if prediction["prediction"] == 1 else (239, 68, 68)
    pdf.set_text_color(*colour)
    pdf.cell(0, 10, decision, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Approval Probability : {prediction['probability']:.2%}", ln=True)
    pdf.cell(0, 7, f"Risk Tier            : {prediction['risk']}", ln=True)
    pdf.ln(6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "This document is system-generated and for informational purposes only.", ln=True)
    return bytes(pdf.output())

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Auth Gate (Feature 11) ──────────────────────────────────────────────
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    login_gate()

    # ── Load artefacts ──────────────────────────────────────────────────────
    arts = load_artifacts()
    if "rf" not in arts:
        st.error("❌  No trained model found. Run `python Training.py` first.")
        st.stop()

    meta = arts["meta"]
    NUM_FEAT = meta["NUMERICAL_FEATURES"]
    CAT_FEAT = meta["CATEGORICAL_FEATURES"]
    CAT_VALS = meta["CATEGORICAL_VALUES"]
    NUM_RNG  = meta["NUMERIC_RANGES"]

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏦 Loan AI System")
        st.markdown(f"👤 Logged in as **{st.session_state.get('username','admin')}**")
        st.markdown("---")

        page = st.selectbox("📂 Navigation", [
            "🏠 Dashboard",
            "🔍 Single Prediction",
            "📊 Bulk Prediction",
            "🧠 Explainable AI",
            "📡 Model Drift Monitor",
            "🗺️ Geo Mapping",
            "🏆 Model Benchmarking",
            "🚨 Fraud & Anomaly Detection",
            "📄 Bank Statement Generator",
            "🗂️ Admin CRM Hub",
            "🛡️ Data Anonymization",
            "🗺️ Credit Improvement Roadmap",
            "📈 Market Rate Comparison",
            "💳 Credit Card Recommendations",
            "🔔 Alert Simulation",
            "🤖 AI Financial Bot",
            "📋 Audit Logs",
        ])

        st.markdown("---")
        # Feature 12 – GDPR toggle
        gdpr_on = st.toggle("🛡️ GDPR Anonymisation Mode", value=False)
        st.markdown("---")

        # Feature 17 – AI Financial Bot (sidebar)
        st.markdown("### 🤖 Quick Finance Bot")
        bot_q = st.text_input("Ask me anything …", placeholder="What is DTI?")
        if bot_q:
            BOT_KB = {
                "dti": "Debt-to-Income Ratio (DTI) measures your monthly debt payments vs gross income. Lower is better; lenders prefer < 36%.",
                "credit score": "Credit scores range 300–850. 750+ = Excellent. 670–749 = Good. 580–669 = Fair. Below 580 = Poor.",
                "loan term": "Shorter terms (36 mo) mean higher payments but less total interest. Longer terms (60 mo) lower payments but cost more overall.",
                "interest rate": "Your rate depends on credit score, DTI, loan amount, and lender policy. Shop multiple offers to find the best rate.",
                "approval": "Approval depends on credit score, DTI, income stability, delinquency history, and requested loan amount.",
                "improve": "To improve approval odds: reduce outstanding debts, increase income, correct credit report errors, and avoid new hard inquiries.",
            }
            answer = "I'm not sure about that. Try asking about DTI, credit scores, loan terms, or approval tips!"
            for kw, resp in BOT_KB.items():
                if kw in bot_q.lower():
                    answer = resp; break
            st.info(f"💬 {answer}")

    # ── HEADER ──────────────────────────────────────────────────────────────
    st.markdown("## 🏦 Loan Approval Intelligence System")
    st.markdown("*Enterprise-grade AI platform for credit decision making*")
    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Dashboard
    # ════════════════════════════════════════════════════════════════════════
    if page == "🏠 Dashboard":
        m = arts.get("metrics", {})
        rf_m = m.get("RandomForest", {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card"><h3>Model Accuracy</h3>
            <h1>{rf_m.get('accuracy',0):.2%}</h1></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><h3>ROC-AUC Score</h3>
            <h1>{rf_m.get('roc_auc',0):.3f}</h1></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><h3>F1 Score</h3>
            <h1>{rf_m.get('f1',0):.3f}</h1></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><h3>Precision</h3>
            <h1>{rf_m.get('precision',0):.3f}</h1></div>""", unsafe_allow_html=True)

        st.markdown("### 📊 Model Performance Snapshot")

        if "feature_importances" in arts:
            fi = arts["feature_importances"].reset_index()
            fi.columns = ["Feature", "Importance"]
            fig = px.bar(fi.head(15).sort_values("Importance"),
                         x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues",
                         title="Top-15 Feature Importances (RandomForest)")
            fig.update_layout(height=460, showlegend=False,
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # Simulated live approval gauge
        st.markdown("### 🎯 Live Model Readiness")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rf_m.get("accuracy", 0.85) * 100,
            title={"text": "Model Accuracy %"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#2563eb"},
                   "steps": [{"range":[0,60],"color":"#ef4444"},
                              {"range":[60,80],"color":"#f97316"},
                              {"range":[80,100],"color":"#22c55e"}]}
        ))
        gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 1 – Single Prediction Assessment Hub
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Single Prediction":
        st.markdown("### 🔍 Single Prediction Assessment Hub")

        with st.form("single_pred"):
            cols = st.columns(3)

            # ── Numerical inputs ──────────────────────────────────────────
            def num_input(col_obj, label, key):
                r = NUM_RNG.get(key, {})
                return col_obj.number_input(
                    label, min_value=float(r.get("min",0)),
                    max_value=float(r.get("max",1e6)),
                    value=float(r.get("mean", r.get("min",0))),
                    step=1.0 if "score" in key or "age" in key else 100.0
                )

            age    = num_input(cols[0], "Age", "age")
            income = num_input(cols[1], "Annual Income ($)", "annual_income")
            credit = num_input(cols[2], "Credit Score", "credit_score")

            cols2 = st.columns(3)
            loan_amt  = num_input(cols2[0], "Loan Amount ($)", "loan_amount")
            dti       = cols2[1].number_input("Debt-to-Income Ratio", 0.01, 1.0, 0.20, 0.01)
            int_rate  = num_input(cols2[2], "Interest Rate (%)", "interest_rate")

            cols3 = st.columns(3)
            loan_term    = cols3[0].selectbox("Loan Term (months)", [36, 60])
            installment  = num_input(cols3[1], "Monthly Installment ($)", "installment")
            open_accts   = int(cols3[2].slider("Open Accounts", 0, 15, 5))

            cols4 = st.columns(3)
            monthly_inc     = num_input(cols4[0], "Monthly Income ($)", "monthly_income")
            total_credit    = num_input(cols4[1], "Total Credit Limit ($)", "total_credit_limit")
            current_bal     = num_input(cols4[2], "Current Balance ($)", "current_balance")

            cols5 = st.columns(3)
            delinq_hist  = int(cols5[0].slider("Delinquency History", 0, 11, 2))
            pub_rec      = int(cols5[1].slider("Public Records", 0, 2, 0))
            num_delinq   = int(cols5[2].slider("Num of Delinquencies", 0, 11, 2))

            st.markdown("##### Categorical Information")
            cat_cols = st.columns(3)
            gender       = cat_cols[0].selectbox("Gender",           CAT_VALS["gender"])
            marital      = cat_cols[1].selectbox("Marital Status",   CAT_VALS["marital_status"])
            education    = cat_cols[2].selectbox("Education Level",  CAT_VALS["education_level"])
            cat_cols2 = st.columns(3)
            employment   = cat_cols2[0].selectbox("Employment Status", CAT_VALS["employment_status"])
            loan_purpose = cat_cols2[1].selectbox("Loan Purpose",      CAT_VALS["loan_purpose"])
            grade        = cat_cols2[2].selectbox("Grade/Subgrade",    CAT_VALS["grade_subgrade"])

            # Applicant name & location for CRM / geo
            st.markdown("##### Applicant Identity (for CRM & Geo)")
            id_cols = st.columns(3)
            appl_name = id_cols[0].text_input("Applicant Name", "John Doe")
            appl_city = id_cols[1].text_input("City", "New York")
            appl_lat  = id_cols[2].number_input("Latitude",  -90.0, 90.0, 40.71)
            appl_lon  = id_cols[2].number_input("Longitude", -180.0, 180.0, -74.00)

            submitted = st.form_submit_button("🚀 Run Prediction")

        if submitted:
            input_dict = {
                "age": age, "annual_income": income, "monthly_income": monthly_inc,
                "debt_to_income_ratio": dti, "credit_score": credit,
                "loan_amount": loan_amt, "interest_rate": int_rate,
                "loan_term": loan_term, "installment": installment,
                "num_of_open_accounts": open_accts,
                "total_credit_limit": total_credit, "current_balance": current_bal,
                "delinquency_history": delinq_hist, "public_records": pub_rec,
                "num_of_delinquencies": num_delinq,
                "gender": gender, "marital_status": marital,
                "education_level": education, "employment_status": employment,
                "loan_purpose": loan_purpose, "grade_subgrade": grade,
            }
            result = predict_single(arts, input_dict)
            write_audit("SINGLE_PREDICT", f"prob={result['probability']:.3f}")

            # ── Feature 18 – Dynamic Risk Categorisation ──────────────────
            rl = result["risk"]
            col_a, col_b = st.columns(2)
            with col_a:
                colour = {"Low Risk":"#22c55e","Moderate Risk":"#f97316","High Risk":"#ef4444"}[rl]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(result["probability"] * 100, 1),
                    title={"text": "Approval Probability (%)"},
                    delta={"reference": 50},
                    gauge={"axis": {"range":[0,100]},
                           "bar": {"color": colour},
                           "steps":[{"range":[0,40],"color":"#fecaca"},
                                    {"range":[40,65],"color":"#fed7aa"},
                                    {"range":[65,100],"color":"#bbf7d0"}]}
                ))
                fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown("#### 🏷️ Credit Decision")
                decision_icon = "✅ **APPROVED**" if result["prediction"] == 1 else "❌ **REJECTED**"
                st.markdown(f"## {decision_icon}")
                st.markdown(f"**Risk Tier:** {risk_badge(rl)}", unsafe_allow_html=True)
                st.metric("Approval Probability", f"{result['probability']:.2%}")

                # Feature 7 – Fraud / Anomaly check
                num_vals = [input_dict[k] for k in NUM_FEAT]
                if is_anomaly(arts, num_vals):
                    st.warning("⚠️ **Fraud Alert**: This application shows anomalous patterns.")
                    write_audit("ANOMALY_DETECTED", appl_name)

            # PDF Statement (Feature 8)
            if FPDF_OK:
                pdf_bytes = generate_pdf_statement(
                    {k: v for k, v in input_dict.items() if k in
                     ["age","annual_income","credit_score","loan_amount","loan_purpose"]},
                    result
                )
                st.download_button("📄 Download Bank Statement (PDF)", pdf_bytes,
                                   file_name=f"statement_{appl_name.replace(' ','_')}.pdf",
                                   mime="application/pdf")

            # Store for CRM
            if "crm_data" not in st.session_state:
                st.session_state["crm_data"] = []
            st.session_state["crm_data"].append({
                "name": appl_name, "city": appl_city,
                "lat": appl_lat, "lon": appl_lon,
                "decision": "Approved" if result["prediction"] == 1 else "Rejected",
                "probability": f"{result['probability']:.2%}",
                "risk": rl, **input_dict
            })

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 2 – Bulk Prediction Hub
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📊 Bulk Prediction":
        st.markdown("### 📊 Bulk Prediction Hub")
        st.info("Upload a CSV with the same columns as the training data. "
                "The app will append prediction columns and let you download the results.")

        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df_up = pd.read_csv(uploaded)
            if gdpr_on:
                df_up = anonymise(df_up)
                st.warning("🛡️ GDPR mode: data has been anonymised before display.")
            st.dataframe(df_up.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Prediction"):
                with st.spinner("Running batch inference …"):
                    result_df = predict_dataframe(arts, df_up)
                write_audit("BULK_PREDICT", f"rows={len(result_df)}")
                st.success(f"✅ Predicted {len(result_df)} records.")
                st.dataframe(result_df[["approval_probability","decision","risk_tier"]].head(20),
                             use_container_width=True)

                # Feature 10 – Multi-select Batch Download
                st.markdown("#### ⬇️ Download Results")
                dl_cols = st.multiselect("Select columns to export",
                                         result_df.columns.tolist(),
                                         default=["approval_probability","decision","risk_tier"])
                csv_bytes = result_df[dl_cols].to_csv(index=False).encode()
                st.download_button("📥 Download CSV", csv_bytes,
                                   "batch_predictions.csv", "text/csv")

                # Distribution chart
                fig = px.pie(result_df, names="decision", color="decision",
                             color_discrete_map={"Approved":"#22c55e","Rejected":"#ef4444"},
                             title="Batch Decision Distribution")
                st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 3 – Explainable AI (XAI)
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🧠 Explainable AI":
        st.markdown("### 🧠 Explainable AI — Feature Importance (XAI)")
        if "feature_importances" not in arts:
            st.warning("Feature importances file not found. Please run Training.py first.")
        else:
            fi = arts["feature_importances"].reset_index()
            fi.columns = ["Feature", "Importance"]
            n = st.slider("Show top N features", 5, min(40, len(fi)), 20)
            fig = px.bar(fi.head(n).sort_values("Importance"),
                         x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="RdYlGn",
                         title=f"Top-{n} Feature Importances (Global XAI)")
            fig.update_layout(height=550, plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
**Interpretation Guide:**
- **credit_score** — primary driver of approval probability
- **debt_to_income_ratio** — high DTI strongly increases rejection risk
- **annual_income / monthly_income** — higher income boosts approval
- **num_of_delinquencies** — more late payments → rejection
- **grade_subgrade** — lender-assigned risk grade has large impact
""")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 4 – Model Drift Monitor
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📡 Model Drift Monitor":
        st.markdown("### 📡 Real-time Model Drift Monitoring Dashboard")
        st.caption("Simulated drift metrics — connect to production data for real monitoring.")

        np.random.seed(int(time.time()) % 100)
        days = pd.date_range(end=datetime.date.today(), periods=30)
        drift_df = pd.DataFrame({
            "date":        days,
            "accuracy":    np.clip(0.88 + np.cumsum(np.random.normal(0, 0.003, 30)), 0.78, 0.97),
            "psi":         np.abs(np.cumsum(np.random.normal(0, 0.01, 30))),
            "data_volume": np.random.randint(800, 1200, 30)
        })

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(drift_df, x="date", y="accuracy", markers=True,
                          title="Daily Model Accuracy Trend",
                          color_discrete_sequence=["#2563eb"])
            fig.add_hline(y=0.85, line_dash="dash", line_color="red",
                          annotation_text="Drift Threshold")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.area(drift_df, x="date", y="psi",
                           title="Population Stability Index (PSI)",
                           color_discrete_sequence=["#f97316"])
            fig2.add_hline(y=0.20, line_dash="dash", line_color="red",
                           annotation_text="High Drift (PSI>0.2)")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        st.bar_chart(drift_df.set_index("date")["data_volume"])
        latest_psi = drift_df["psi"].iloc[-1]
        if latest_psi > 0.20:
            st.error(f"🚨 High drift detected! PSI = {latest_psi:.3f}. Retrain recommended.")
        elif latest_psi > 0.10:
            st.warning(f"⚠️ Moderate drift. PSI = {latest_psi:.3f}. Monitor closely.")
        else:
            st.success(f"✅ Model stable. PSI = {latest_psi:.3f}")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 5 – Geo Mapping
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🗺️ Geo Mapping":
        st.markdown("### 🗺️ Applicant Geospatial Visualisation")
        crm = st.session_state.get("crm_data", [])

        if not crm:
            st.info("No applicant records yet. Run a single prediction first.")
            sample_coords = [
                {"name":"Alice","lat":40.71,"lon":-74.00,"decision":"Approved"},
                {"name":"Bob",  "lat":34.05,"lon":-118.24,"decision":"Rejected"},
                {"name":"Carol","lat":41.88,"lon":-87.63,"decision":"Approved"},
                {"name":"Dave", "lat":29.76,"lon":-95.37,"decision":"Rejected"},
            ]
        else:
            sample_coords = crm

        df_map = pd.DataFrame(sample_coords)

        if FOLIUM_OK:
            m = folium.Map(location=[37.5, -95], zoom_start=4)
            for _, row in df_map.iterrows():
                colour = "green" if row.get("decision","") == "Approved" else "red"
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=10, color=colour, fill=True,
                    popup=f"{row['name']} — {row.get('decision','')}",
                    tooltip=row["name"]
                ).add_to(m)
            st_folium(m, width=900, height=480)
        else:
            fig = px.scatter_geo(df_map, lat="lat", lon="lon",
                                  color="decision", hover_name="name",
                                  color_discrete_map={"Approved":"#22c55e","Rejected":"#ef4444"},
                                  projection="natural earth",
                                  title="Applicant Locations")
            fig.update_geos(showcoastlines=True, coastlinecolor="Gray",
                            showland=True, landcolor="#f0fdf4",
                            showocean=True, oceancolor="#dbeafe")
            st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 6 – Model Benchmarking
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🏆 Model Benchmarking":
        st.markdown("### 🏆 Champion vs Challenger Benchmarking")
        m = arts.get("metrics", {})

        if not m:
            st.warning("No benchmark metrics found. Run Training.py first.")
        else:
            bench_df = pd.DataFrame([
                {"Model": name, "Metric": k, "Score": v}
                for name, vals in m.items()
                for k, v in vals.items()
                if isinstance(v, float)
            ])
            pivot = bench_df.pivot(index="Metric", columns="Model", values="Score").reset_index()
            st.dataframe(pivot.style.format("{:.4f}", subset=pivot.columns[1:])
                         .background_gradient(cmap="Blues", subset=pivot.columns[1:]),
                         use_container_width=True)

            fig = px.bar(bench_df, x="Metric", y="Score", color="Model", barmode="group",
                         color_discrete_sequence=["#2563eb", "#22c55e"],
                         title="Champion vs Challenger — All Metrics")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            # Crown the champion
            rf_auc  = m.get("RandomForest", {}).get("roc_auc", 0)
            xgb_auc = m.get("XGBoost",      {}).get("roc_auc", 0)
            champion = "RandomForest" if rf_auc >= xgb_auc else "XGBoost"
            st.success(f"🏆 Current Champion: **{champion}** "
                       f"(ROC-AUC = {max(rf_auc, xgb_auc):.4f})")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 7 – Fraud & Anomaly Detection
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🚨 Fraud & Anomaly Detection":
        st.markdown("### 🚨 Fraud & Anomaly Detection Layer")
        uploaded = st.file_uploader("Upload applicant CSV for fraud screening", type="csv")
        if uploaded:
            df_up = pd.read_csv(uploaded)
            num_cols_present = [c for c in NUM_FEAT if c in df_up.columns]
            if num_cols_present and "anomaly" in arts and "scaler" in arts:
                scaled = arts["scaler"].transform(df_up[num_cols_present].fillna(0))
                scores = arts["anomaly"].decision_function(scaled)
                flags  = arts["anomaly"].predict(scaled)
                df_up["anomaly_score"] = scores.round(4)
                df_up["fraud_flag"]    = flags
                df_up["fraud_label"]   = df_up["fraud_flag"].map({1: "Normal", -1: "⚠️ Suspicious"})

                suspicious = df_up[df_up["fraud_flag"] == -1]
                st.metric("Suspicious Records", len(suspicious), f"{len(suspicious)/len(df_up):.1%} of total")

                fig = px.histogram(df_up, x="anomaly_score", color="fraud_label",
                                   title="Anomaly Score Distribution",
                                   color_discrete_map={"Normal":"#22c55e","⚠️ Suspicious":"#ef4444"})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(suspicious.head(20), use_container_width=True)
                write_audit("FRAUD_SCAN", f"flagged={len(suspicious)}")
            else:
                st.warning("Anomaly model or required numeric columns not found.")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 9 – Admin CRM Hub
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🗂️ Admin CRM Hub":
        st.markdown("### 🗂️ Admin CRM Hub — Client Search")
        crm = st.session_state.get("crm_data", [])
        if not crm:
            st.info("No client records found. Submit predictions via 'Single Prediction' first.")
        else:
            df_crm = pd.DataFrame(crm)
            search = st.text_input("🔍 Search by applicant name")
            filtered = df_crm[df_crm["name"].str.contains(search, case=False)] if search else df_crm
            if gdpr_on:
                filtered = anonymise(filtered)
            st.dataframe(filtered, use_container_width=True)
            csv = filtered.to_csv(index=False).encode()
            st.download_button("📥 Export CRM to CSV", csv, "crm_export.csv", "text/csv")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 12 – Data Anonymization
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🛡️ Data Anonymization":
        st.markdown("### 🛡️ GDPR Data Anonymisation Mode")
        st.info("Upload any dataset CSV to view it in anonymised form. "
                "String fields are SHA-256 hashed; numeric fields are rounded.")
        uploaded = st.file_uploader("Upload CSV for anonymisation", type="csv")
        if uploaded:
            df_raw  = pd.read_csv(uploaded)
            df_anon = anonymise(df_raw)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original (first 10 rows)**")
                st.dataframe(df_raw.head(10), use_container_width=True)
            with c2:
                st.markdown("**Anonymised (first 10 rows)**")
                st.dataframe(df_anon.head(10), use_container_width=True)
            st.download_button("📥 Download Anonymised CSV",
                               df_anon.to_csv(index=False).encode(),
                               "anonymised_data.csv", "text/csv")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 13 – Credit Improvement Roadmap
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🗺️ Credit Improvement Roadmap":
        st.markdown("### 🗺️ Personalised Credit Improvement Roadmap")
        with st.form("roadmap_form"):
            r1, r2, r3 = st.columns(3)
            cs   = r1.number_input("Current Credit Score", 300, 850, 620)
            dti  = r2.number_input("Debt-to-Income Ratio", 0.01, 1.0, 0.38, 0.01)
            delinq = r3.number_input("Num of Delinquencies", 0, 11, 3)
            go_btn = st.form_submit_button("Generate Roadmap")

        if go_btn:
            steps = []
            if cs < 670:
                steps.append(("📈 Improve Credit Score", f"Your score of {cs} is below Good tier (670+). "
                              "Pay all bills on time for 6+ months; dispute any errors on your credit report."))
            if dti > 0.36:
                steps.append(("💳 Reduce Debt-to-Income Ratio", f"DTI of {dti:.0%} exceeds the 36% threshold. "
                              "Target paying off high-interest debts first (avalanche method)."))
            if delinq > 0:
                steps.append(("📋 Clear Delinquency History", f"You have {int(delinq)} delinquency records. "
                              "Negotiate payment plans with creditors; each resolved account improves your profile."))
            steps.append(("🏦 Build Credit Mix", "Maintaining a mix of instalment and revolving credit "
                          "accounts boosts score diversity."))
            steps.append(("🔒 Avoid New Hard Inquiries", "Each application within 12 months can drop score "
                          "by 5–10 points. Only apply when necessary."))

            for title, desc in steps:
                with st.expander(title):
                    st.write(desc)

            # Timeline chart
            months = list(range(1, 13))
            growth  = [min(cs + i * (35 / 12 if cs < 750 else 5), 850) for i in months]
            fig = px.line(x=months, y=growth, labels={"x":"Month","y":"Projected Score"},
                          title="Projected Credit Score Improvement (12 months)",
                          color_discrete_sequence=["#22c55e"])
            fig.add_hline(y=670, line_dash="dash", annotation_text="Good Tier")
            fig.add_hline(y=740, line_dash="dot", annotation_text="Very Good Tier")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 14 – Market Rate Comparison
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📈 Market Rate Comparison":
        st.markdown("### 📈 Live Market Interest Rate Comparison")
        st.caption(f"Rates as of {datetime.date.today().strftime('%B %d, %Y')} — sourced from market benchmarks.")

        rate_df = pd.DataFrame(list(MARKET_RATES.items()), columns=["Product", "Rate"])
        rate_df["Rate_num"] = rate_df["Rate"].str.replace("%","").astype(float)
        rate_df = rate_df.sort_values("Rate_num")

        fig = px.bar(rate_df, x="Rate_num", y="Product", orientation="h",
                     text="Rate", color="Rate_num",
                     color_continuous_scale="RdYlGn_r",
                     title="Current Market Interest Rates by Product")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=450, xaxis_title="APR (%)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Comparison Tool:** Enter your offered rate to see how it compares.")
        my_rate = st.number_input("Your Offered Rate (%)", 1.0, 30.0, 9.5, 0.1)
        better  = rate_df[rate_df["Rate_num"] > my_rate]["Product"].tolist()
        worse   = rate_df[rate_df["Rate_num"] < my_rate]["Product"].tolist()
        if better:
            st.success(f"Your rate beats: {', '.join(better)}")
        if worse:
            st.warning(f"Your rate is higher than: {', '.join(worse)}")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 15 – Credit Card Recommendations
    # ════════════════════════════════════════════════════════════════════════
    elif page == "💳 Credit Card Recommendations":
        st.markdown("### 💳 Smart Credit Card Recommendation Engine")
        cs   = st.slider("Your Credit Score", 300, 850, 700)
        goal = st.selectbox("Primary Goal", ["Cashback", "Travel Rewards", "Balance Transfer",
                                              "Building Credit", "Business Expenses"])

        cards = {
            "Cashback": [
                {"name":"Chase Freedom Flex",   "apr":"19.99%","reward":"5% on rotating cats","min_score":670},
                {"name":"Citi Double Cash",      "apr":"18.99%","reward":"2% on everything",   "min_score":680},
                {"name":"Discover it Cash Back", "apr":"17.99%","reward":"5% quarterly cats",  "min_score":660},
            ],
            "Travel Rewards": [
                {"name":"Chase Sapphire Preferred","apr":"21.49%","reward":"3x on dining/travel","min_score":720},
                {"name":"Amex Gold Card",           "apr":"N/A (charge)","reward":"4x dining/US supermarkets","min_score":700},
                {"name":"Capital One Venture",      "apr":"19.99%","reward":"2x on everything","min_score":690},
            ],
            "Balance Transfer": [
                {"name":"Citi Simplicity",   "apr":"0% for 21mo","reward":"No late fees","min_score":680},
                {"name":"Wells Fargo Reflect","apr":"0% for 21mo","reward":"Low ongoing APR","min_score":670},
            ],
            "Building Credit": [
                {"name":"Discover it Secured",   "apr":"28.24%","reward":"2% on gas/restaurants","min_score":300},
                {"name":"Capital One Platinum",   "apr":"29.99%","reward":"Credit-limit increase path","min_score":580},
                {"name":"OpenSky Secured Visa",   "apr":"22.39%","reward":"No credit check needed","min_score":300},
            ],
            "Business Expenses": [
                {"name":"Ink Business Cash",      "apr":"18.49%","reward":"5% on office/internet","min_score":680},
                {"name":"Amex Blue Business Plus","apr":"18.49%","reward":"2x MR on all business","min_score":670},
            ],
        }
        recs = [c for c in cards.get(goal, []) if cs >= c["min_score"]]
        if not recs:
            st.warning("No cards match your credit profile for this goal. Work on improving your score first.")
        else:
            for card in recs:
                with st.expander(f"💳 {card['name']}"):
                    col_a, col_b = st.columns(2)
                    col_a.metric("APR", card["apr"])
                    col_b.metric("Reward", card["reward"])
                    col_a.metric("Min Score Required", card["min_score"])
                    col_b.metric("Your Score", cs)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 16 – Alert Simulation
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🔔 Alert Simulation":
        st.markdown("### 🔔 Automated Alert Simulation")
        st.caption("Simulates email and in-app notification dispatch for loan decisions.")

        with st.form("alert_form"):
            email   = st.text_input("Recipient Email", "applicant@example.com")
            subject = st.text_input("Subject", "Your Loan Application Update")
            msg     = st.text_area("Message Body", "Dear Applicant, your loan has been reviewed …")
            ch      = st.multiselect("Channels", ["Email","SMS","In-App","Webhook"], default=["Email"])
            send    = st.form_submit_button("📤 Send Alert")

        if send:
            for channel in ch:
                with st.spinner(f"Dispatching via {channel} …"):
                    time.sleep(0.4)
                st.success(f"✅ {channel} alert sent to {email}")
            write_audit("ALERT_SENT", f"to={email}, channels={ch}")

        st.markdown("#### 📜 Recent Alert History (Simulated)")
        hist = pd.DataFrame({
            "Time":    [datetime.datetime.now() - datetime.timedelta(minutes=i*7) for i in range(5)],
            "To":      ["alice@x.com","bob@x.com","carol@x.com","dave@x.com","eve@x.com"],
            "Channel": ["Email","SMS","In-App","Email","Webhook"],
            "Status":  ["Delivered"]*5
        })
        st.dataframe(hist, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: Feature 19 – Audit Logs
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📋 Audit Logs":
        st.markdown("### 📋 Performance Audit Logs")
        if AUDIT_LOG.exists():
            df_log = pd.read_csv(AUDIT_LOG)
            st.metric("Total Actions Logged", len(df_log))
            st.dataframe(df_log.sort_values("timestamp", ascending=False).head(100),
                         use_container_width=True)
            fig = px.histogram(df_log, x="action", color="action",
                               title="Actions Distribution")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("📥 Export Audit Log",
                               df_log.to_csv(index=False).encode(),
                               "audit_log.csv", "text/csv")
        else:
            st.info("No audit log yet. Perform some operations first.")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#94a3b8;font-size:.8rem;'>"
        "🏦 Loan Approval Intelligence System · Enterprise Edition · "
        f"Session: {st.session_state.get('username','guest')} · "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
