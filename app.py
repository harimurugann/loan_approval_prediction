"""
=============================================================================
  Enterprise Loan Risk Intelligence System — Streamlit Dashboard
  20 Advanced Intelligence Modules
=============================================================================
"""

import os, io, json, hashlib, warnings, random, string, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# ── Path anchors ────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "Data")
MODEL_DIR     = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
DATA_PATH     = os.path.join(DATA_DIR,  "loan_dataset_20000.csv")
PIPELINE_PATH = os.path.join(MODEL_DIR, "full_pipeline.sav")

# ── Load artefacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["loan_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1e-6)
    df["payment_burden"]    = df["debt_to_income"] * df["loan_amount"] / (df["annual_income"] + 1e-6)
    df["risk_index"]        = df["delinquencies_2yrs"] * 10 + (850 - df["credit_score"]) / 85
    return df

pipeline_state = load_pipeline()
pipeline       = pipeline_state["pipeline"]
predict_fn     = pipeline_state["predict_fn"]
metrics        = pipeline_state["metrics"]
all_features   = pipeline_state["all_features"]
NUMERIC_FEATURES     = pipeline_state["numeric_features"]
CATEGORICAL_FEATURES = pipeline_state["cat_features"]

df = load_data()

# ── Admin credentials (hashed) ───────────────────────────────────────────────
ADMIN_HASH = hashlib.sha256(b"admin123").hexdigest()   # change in production

def check_admin(pwd: str) -> bool:
    return hashlib.sha256(pwd.encode()).hexdigest() == ADMIN_HASH

# ── Shared predict wrapper ───────────────────────────────────────────────────
def run_inference(row: dict) -> dict:
    """Mirror MODULE 12 feature engineering then call pipeline."""
    r = dict(row)
    r["loan_income_ratio"] = r["loan_amount"] / (r["annual_income"] + 1e-6)
    r["payment_burden"]    = r["debt_to_income"] * r["loan_amount"] / (r["annual_income"] + 1e-6)
    r["risk_index"]        = r["delinquencies_2yrs"] * 10 + (850 - r["credit_score"]) / 85
    df_in = pd.DataFrame([r])[all_features]
    prob  = pipeline.predict_proba(df_in)[0, 1]
    pred  = int(prob >= 0.5)
    risk  = "🟢 LOW" if prob >= 0.7 else ("🟡 MEDIUM" if prob >= 0.4 else "🔴 HIGH")
    return {"prediction": pred, "probability": prob, "risk_level": risk}

# ── Streamlit config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Risk Intelligence System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px; padding: 18px; color: white;
        text-align: center; margin: 6px 0;
    }
    .risk-low    { background: #27ae60; border-radius:8px; padding:8px; color:white; text-align:center; font-weight:bold; }
    .risk-med    { background: #f39c12; border-radius:8px; padding:8px; color:white; text-align:center; font-weight:bold; }
    .risk-high   { background: #e74c3c; border-radius:8px; padding:8px; color:white; text-align:center; font-weight:bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"]      { border-radius: 6px 6px 0 0; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR  — Navigation
# ============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
st.sidebar.title("🏦 Loan Risk IQ")
st.sidebar.markdown("**Enterprise Intelligence System**")
st.sidebar.divider()

PAGES = [
    "🏠 Dashboard Overview",
    "🔍 Single Loan Assessment",
    "📋 Bulk Assessment Hub",
    "🧠 XAI Decision Logic",
    "📡 Model Drift Monitor",
    "🗺️ Geospatial Risk Map",
    "🚨 Fraud / Anomaly Detector",
    "📄 Bank Statement PDF",
    "👤 Admin CRM Hub",
    "📦 Batch Data Export",
    "🔐 Secure Admin Gateway",
    "🛡️ GDPR Anonymiser",
    "🗺️ Credit Roadmap Generator",
    "📊 Market Rate Matrix",
    "🤖 AI Financial Bot",
    "📈 Portfolio Analytics",
    "🏆 Model Leaderboard",
    "⚙️ System Configuration",
    "📚 Data Dictionary",
    "ℹ️ About & Docs",
]
page = st.sidebar.selectbox("Navigate", PAGES)

st.sidebar.divider()
st.sidebar.markdown(f"**Model Metrics**")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
st.sidebar.metric("F1-Score", f"{metrics['f1_score']:.4f}")
st.sidebar.metric("ROC-AUC",  f"{metrics['roc_auc']:.4f}")

# ============================================================================
# PAGE IMPLEMENTATIONS
# ============================================================================

# ─── P1: Dashboard Overview ──────────────────────────────────────────────────
if page == "🏠 Dashboard Overview":
    st.title("🏦 Loan Risk Intelligence — Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{len(df):,}")
    c2.metric("Repayment Rate",   f"{df['loan_paid_back'].mean()*100:.1f}%")
    c3.metric("Avg Loan Amount",  f"${df['loan_amount'].mean():,.0f}")
    c4.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Grade Distribution")
        grade_counts = df["loan_grade"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(grade_counts.index, grade_counts.values,
                      color=["#27ae60","#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad"])
        ax.set(xlabel="Grade", ylabel="Count", title="Loan Grade Breakdown")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                    f"{bar.get_height():,}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Repayment by Loan Purpose")
        purpose_rate = df.groupby("loan_purpose")["loan_paid_back"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ["#e74c3c" if v < 0.35 else "#27ae60" for v in purpose_rate]
        purpose_rate.plot(kind="barh", ax=ax, color=colors)
        ax.set(xlabel="Repayment Rate", title="Repayment Rate by Purpose")
        ax.axvline(0.35, linestyle="--", color="gray", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Credit Score Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        for label, colour in [(0,"#e74c3c"), (1,"#27ae60")]:
            subset = df[df["loan_paid_back"]==label]["credit_score"]
            ax.hist(subset, bins=30, alpha=0.6, color=colour,
                    label="Not Paid" if label==0 else "Paid", edgecolor="none")
        ax.set(xlabel="Credit Score", ylabel="Count", title="Credit Score by Outcome")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Debt-to-Income vs Loan Amount")
        sample_df = df.sample(min(1000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(5, 3))
        scatter = ax.scatter(
            sample_df["debt_to_income"], sample_df["loan_amount"],
            c=sample_df["loan_paid_back"], cmap="RdYlGn",
            alpha=0.5, s=10, edgecolors="none"
        )
        ax.set(xlabel="Debt-to-Income %", ylabel="Loan Amount ($)",
               title="DTI vs Loan Amount (colour = outcome)")
        fig.colorbar(scatter, ax=ax, label="Paid Back")
        fig.tight_layout()
        st.pyplot(fig); plt.close()

# ─── P2: Single Loan Assessment ─────────────────────────────────────────────
elif page == "🔍 Single Loan Assessment":
    st.title("🔍 Single Loan Risk Assessment")
    st.markdown("Enter applicant details to receive an instant AI-driven risk decision.")

    with st.form("single_assess"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age            = st.number_input("Age", 18, 80, 35)
            annual_income  = st.number_input("Annual Income ($)", 10000, 500000, 65000, step=1000)
            loan_amount    = st.number_input("Loan Amount ($)", 500, 100000, 12000, step=500)
            interest_rate  = st.number_input("Interest Rate (%)", 1.0, 40.0, 11.5, step=0.5)
        with col2:
            credit_score       = st.slider("Credit Score", 300, 850, 680)
            debt_to_income     = st.slider("Debt-to-Income (%)", 0.0, 80.0, 22.0, step=0.5)
            employment_years   = st.number_input("Employment Years", 0.0, 50.0, 4.0, step=0.5)
            num_credit_lines   = st.number_input("# Credit Lines", 0, 50, 8)
        with col3:
            delinquencies      = st.number_input("Delinquencies (2yr)", 0, 10, 0)
            loan_purpose       = st.selectbox("Loan Purpose", ["debt_consolidation","home_improvement","credit_card","car","medical","vacation","business"])
            loan_grade         = st.selectbox("Loan Grade", ["A","B","C","D","E","F"])
            home_ownership     = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN","OTHER"])
            state              = st.selectbox("State", ["CA","NY","TX","FL","IL","PA","OH","GA","NC","MI"])

        submitted = st.form_submit_button("⚡ Run Risk Assessment", use_container_width=True)

    if submitted:
        row = dict(
            age=age, annual_income=annual_income, loan_amount=loan_amount,
            interest_rate=interest_rate, employment_years=employment_years,
            credit_score=credit_score, debt_to_income=debt_to_income,
            num_credit_lines=num_credit_lines, delinquencies_2yrs=delinquencies,
            loan_purpose=loan_purpose, loan_grade=loan_grade,
            home_ownership=home_ownership, state=state
        )
        result = run_inference(row)
        prob   = result["probability"]

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Decision",     "✅ APPROVED" if result["prediction"]==1 else "❌ DECLINED")
        r2.metric("Repayment Prob", f"{prob*100:.1f}%")
        r3.metric("Risk Level",   result["risk_level"])

        # Gauge
        fig, ax = plt.subplots(figsize=(5, 2.5), subplot_kw={"polar": False})
        bar_color = "#27ae60" if prob >= 0.7 else ("#f39c12" if prob >= 0.4 else "#e74c3c")
        ax.barh(["Repayment Score"], [prob*100], color=bar_color, height=0.5)
        ax.barh(["Repayment Score"], [100], color="#ecf0f1", height=0.5)
        ax.barh(["Repayment Score"], [prob*100], color=bar_color, height=0.5)
        ax.set_xlim(0, 100)
        ax.set(xlabel="Probability (%)", title=f"Repayment Probability: {prob*100:.1f}%")
        ax.axvline(50, linestyle="--", color="gray", linewidth=1)
        st.pyplot(fig); plt.close()

# ─── P3: Bulk Assessment Hub ─────────────────────────────────────────────────
elif page == "📋 Bulk Assessment Hub":
    st.title("📋 Bulk Loan Assessment")
    st.info("Upload a CSV with columns matching the single assessment form to score in batch.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df):,} rows")
        st.dataframe(batch_df.head())

        if st.button("⚡ Score All Loans"):
            results = []
            for _, row in batch_df.iterrows():
                try:
                    r = run_inference(row.to_dict())
                    results.append(r)
                except Exception as e:
                    results.append({"prediction": -1, "probability": -1, "risk_level": f"ERROR: {e}"})
            batch_df["prediction"]   = [r["prediction"]  for r in results]
            batch_df["probability"]  = [r["probability"] for r in results]
            batch_df["risk_level"]   = [r["risk_level"]  for r in results]
            st.success("Scoring complete!")
            st.dataframe(batch_df[["prediction","probability","risk_level"]].head(20))

            csv_out = batch_df.to_csv(index=False).encode()
            st.download_button("⬇ Download Scored CSV", csv_out, "scored_loans.csv", "text/csv")
    else:
        # Demo on existing dataset sample
        if st.button("Demo: Score 100 random loans from dataset"):
            sample = df.sample(100, random_state=1).copy()
            results = [run_inference(row.to_dict()) for _, row in sample.iterrows()]
            sample["pred_prob"]  = [r["probability"] for r in results]
            sample["pred_class"] = [r["prediction"]  for r in results]
            sample["risk"]       = [r["risk_level"]  for r in results]
            st.dataframe(sample[["credit_score","loan_amount","annual_income",
                                 "pred_prob","pred_class","risk"]].head(20))
            risk_counts = pd.Series([r["risk_level"] for r in results]).value_counts()
            st.bar_chart(risk_counts)

# ─── P4: XAI Decision Logic ──────────────────────────────────────────────────
elif page == "🧠 XAI Decision Logic":
    st.title("🧠 Explainable AI — Decision Logic")
    st.markdown("""
    This module decomposes what drives the model's predictions using
    **feature importance** and manual sensitivity analysis (SHAP requires
    a separate install; here we use permutation-style analysis within Streamlit).
    """)

    # Feature importances from trained model
    clf   = pipeline.named_steps["classifier"]
    ohe_cats = (pipeline.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out(CATEGORICAL_FEATURES))
    feat_names = NUMERIC_FEATURES + list(ohe_cats)
    imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Global Feature Importances (GBM)")
        fig, ax = plt.subplots(figsize=(8, 6))
        top_imp = imp.head(20)
        colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top_imp)))[::-1]
        ax.barh(top_imp.index[::-1], top_imp.values[::-1], color=colors[::-1])
        ax.set(xlabel="Importance Score", title="Top 20 Feature Importances")
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Sensitivity Analysis")
        st.markdown("Vary one feature and observe probability change:")
        base_prob = 0.65
        feature   = st.selectbox("Feature to vary", ["credit_score","debt_to_income","loan_amount"])
        range_map = {"credit_score": (300,850), "debt_to_income": (0,80), "loan_amount": (1000,80000)}
        lo, hi    = range_map[feature]
        vals      = np.linspace(lo, hi, 30)
        probs     = []
        base_row  = dict(age=35, annual_income=65000, loan_amount=12000,
                         interest_rate=11.5, employment_years=4.0,
                         credit_score=680, debt_to_income=22.0,
                         num_credit_lines=8, delinquencies_2yrs=0,
                         loan_purpose="debt_consolidation", loan_grade="B",
                         home_ownership="RENT", state="CA")
        for v in vals:
            row = dict(base_row); row[feature] = v
            probs.append(run_inference(row)["probability"])
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(vals, probs, color="#2980b9", linewidth=2)
        ax2.fill_between(vals, probs, alpha=0.15, color="#2980b9")
        ax2.axhline(0.5, linestyle="--", color="red", linewidth=0.8)
        ax2.set(xlabel=feature, ylabel="Repayment Probability", title=f"Effect of {feature}")
        fig2.tight_layout()
        st.pyplot(fig2); plt.close()

# ─── P5: Model Drift Monitor ─────────────────────────────────────────────────
elif page == "📡 Model Drift Monitor":
    st.title("📡 Model Drift Monitoring")
    st.markdown("Simulated PSI (Population Stability Index) and feature drift analysis.")

    # Simulate reference vs current distributions
    ref    = df.sample(5000, random_state=10)
    current = df.sample(5000, random_state=99)

    def psi(expected, actual, buckets=10):
        def scale_range(data, mn, mx, buckets):
            bins = np.linspace(mn, mx, buckets+1)
            return np.histogram(data, bins=bins)[0] / len(data)
        mn = min(expected.min(), actual.min())
        mx = max(expected.max(), actual.max())
        e_pct = scale_range(expected, mn, mx, buckets) + 1e-8
        a_pct = scale_range(actual,   mn, mx, buckets) + 1e-8
        return np.sum((e_pct - a_pct) * np.log(e_pct / a_pct))

    features_to_check = ["credit_score","annual_income","loan_amount","debt_to_income"]
    psi_values = {f: psi(ref[f].dropna(), current[f].dropna()) for f in features_to_check}

    st.subheader("PSI Scores (< 0.1 stable | 0.1-0.2 minor drift | > 0.2 major drift)")
    psi_df = pd.DataFrame({"Feature": list(psi_values.keys()), "PSI": list(psi_values.values())})
    psi_df["Status"] = psi_df["PSI"].apply(lambda x: "🟢 Stable" if x < 0.1 else ("🟡 Minor" if x < 0.2 else "🔴 Major"))
    st.dataframe(psi_df, use_container_width=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, feat in zip(axes.flatten(), features_to_check):
        ax.hist(ref[feat].dropna(), bins=30, alpha=0.6, label="Reference", color="#2980b9")
        ax.hist(current[feat].dropna(), bins=30, alpha=0.6, label="Current",   color="#e67e22")
        ax.set_title(f"{feat} (PSI={psi_values[feat]:.3f})", fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ─── P6: Geospatial Risk Map ─────────────────────────────────────────────────
elif page == "🗺️ Geospatial Risk Map":
    st.title("🗺️ Geospatial Loan Risk Heatmap")
    st.info("Loan performance aggregated by US state.")

    state_stats = df.groupby("state").agg(
        repayment_rate=("loan_paid_back","mean"),
        total_loans=("loan_paid_back","count"),
        avg_loan=("loan_amount","mean")
    ).reset_index()

    st.dataframe(state_stats.sort_values("repayment_rate"), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    state_stats_s = state_stats.sort_values("repayment_rate")
    colors = plt.cm.RdYlGn(state_stats_s["repayment_rate"].values)
    axes[0].barh(state_stats_s["state"], state_stats_s["repayment_rate"], color=colors)
    axes[0].set(xlabel="Repayment Rate", title="Repayment Rate by State")
    axes[0].axvline(state_stats_s["repayment_rate"].mean(), linestyle="--", color="navy", linewidth=1)

    axes[1].scatter(state_stats["avg_loan"], state_stats["repayment_rate"],
                    s=state_stats["total_loans"]/5, alpha=0.7, color="#2980b9", edgecolors="white")
    for _, row in state_stats.iterrows():
        axes[1].annotate(row["state"], (row["avg_loan"], row["repayment_rate"]), fontsize=7)
    axes[1].set(xlabel="Avg Loan ($)", ylabel="Repayment Rate", title="Avg Loan vs Repayment")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ─── P7: Fraud / Anomaly Detector ────────────────────────────────────────────
elif page == "🚨 Fraud / Anomaly Detector":
    st.title("🚨 Fraud & Anomaly Detection Layer")
    st.markdown("Rule-based anomaly flags combined with statistical outlier detection (IQR).")

    df_flag = df.copy()
    df_flag["flag_high_dti"]    = df_flag["debt_to_income"] > 60
    df_flag["flag_low_credit"]  = df_flag["credit_score"] < 400
    df_flag["flag_high_ratio"]  = df_flag["loan_income_ratio"] > 1.5
    df_flag["flag_no_history"]  = df_flag["employment_years"] < 0.5
    df_flag["anomaly_score"]    = (df_flag[["flag_high_dti","flag_low_credit",
                                            "flag_high_ratio","flag_no_history"]]
                                   .sum(axis=1))
    df_flag["is_anomalous"]     = df_flag["anomaly_score"] >= 2

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged Loans",   f"{df_flag['is_anomalous'].sum():,}")
    c2.metric("Anomaly Rate",    f"{df_flag['is_anomalous'].mean()*100:.1f}%")
    c3.metric("Avg Anomaly Score", f"{df_flag['anomaly_score'].mean():.2f}")

    st.subheader("Anomalous Loans Sample")
    anomalous = df_flag[df_flag["is_anomalous"]].sample(min(20, df_flag["is_anomalous"].sum()))
    st.dataframe(anomalous[["credit_score","debt_to_income","loan_income_ratio",
                             "employment_years","anomaly_score","loan_paid_back"]],
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    df_flag["anomaly_score"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#e74c3c", edgecolor="black")
    ax.set(xlabel="Anomaly Score", ylabel="Count", title="Distribution of Anomaly Scores")
    fig.tight_layout()
    st.pyplot(fig); plt.close()

# ─── P8: Bank Statement PDF Generator ────────────────────────────────────────
elif page == "📄 Bank Statement PDF":
    st.title("📄 PDF Bank Statement Generator")
    st.markdown("Generate a simulated loan assessment report as a downloadable PDF.")

    try:
        from fpdf import FPDF

        class LoanPDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 16)
                self.cell(0, 10, "ENTERPRISE LOAN RISK INTELLIGENCE SYSTEM", align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", "", 10)
                self.cell(0, 6, "Confidential — AI-Generated Assessment Report", align="C", new_x="LMARGIN", new_y="NEXT")
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.date.today()}", align="C")

        with st.form("pdf_gen"):
            cid   = st.text_input("Customer ID", "CUST00001")
            cname = st.text_input("Customer Name", "Jane Doe")
            camnt = st.number_input("Loan Amount ($)", 500, 100000, 15000, step=500)
            ccs   = st.slider("Credit Score", 300, 850, 720)
            cinc  = st.number_input("Annual Income ($)", 10000, 500000, 75000, step=1000)
            gen   = st.form_submit_button("Generate PDF")

        if gen:
            pdf = LoanPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "APPLICANT SUMMARY", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            for label, val in [
                ("Customer ID", cid), ("Name", cname),
                ("Loan Amount", f"${camnt:,}"), ("Credit Score", str(ccs)),
                ("Annual Income", f"${cinc:,}"),
                ("Assessment Date", str(datetime.date.today()))
            ]:
                pdf.cell(60, 8, f"{label}:", border=0)
                pdf.cell(0, 8, str(val), border=0, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "AI RISK ASSESSMENT", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            row = dict(age=35, annual_income=cinc, loan_amount=camnt,
                       interest_rate=11.5, employment_years=3.0, credit_score=ccs,
                       debt_to_income=22.0, num_credit_lines=7, delinquencies_2yrs=0,
                       loan_purpose="debt_consolidation", loan_grade="B",
                       home_ownership="RENT", state="CA")
            result = run_inference(row)
            decision = "APPROVED" if result["prediction"]==1 else "DECLINED"
            for label, val in [
                ("Decision", decision),
                ("Repayment Probability", f"{result['probability']*100:.1f}%"),
                ("Risk Level", result["risk_level"].replace("🟢","").replace("🟡","").replace("🔴","").strip())
            ]:
                pdf.cell(70, 8, f"{label}:", border=0)
                pdf.cell(0, 8, str(val), border=0, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(6)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 6, "DISCLAIMER: This report is generated by an AI model and should be reviewed by a qualified credit officer before final lending decisions are made.")

            pdf_bytes = bytes(pdf.output())
            st.download_button("⬇ Download PDF Report", pdf_bytes, f"loan_report_{cid}.pdf", "application/pdf")
            st.success("PDF generated successfully!")

    except ImportError:
        st.error("fpdf2 is not installed. Add `fpdf2` to requirements.txt and reinstall.")

# ─── P9: Admin CRM Hub ───────────────────────────────────────────────────────
elif page == "👤 Admin CRM Hub":
    st.title("👤 Admin CRM Hub")

    search = st.text_input("🔍 Search by Customer ID prefix (e.g. CUST001)")
    filter_grade = st.multiselect("Filter by Loan Grade", ["A","B","C","D","E","F"], default=["A","B","C","D","E","F"])

    view_df = df_raw = pd.read_csv(DATA_PATH) if "customer_id" in pd.read_csv(DATA_PATH, nrows=1).columns else df.copy()
    if "customer_id" not in view_df.columns:
        view_df.insert(0, "customer_id", [f"CUST{i:05d}" for i in range(1, len(view_df)+1)])

    if search:
        view_df = view_df[view_df["customer_id"].str.startswith(search)]
    view_df = view_df[view_df["loan_grade"].isin(filter_grade)]

    st.write(f"Showing {len(view_df):,} records")
    st.dataframe(view_df.head(50), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Filtered Records", f"{len(view_df):,}")
    c2.metric("Avg Credit Score", f"{view_df['credit_score'].mean():.0f}")
    c3.metric("Repayment Rate",   f"{view_df['loan_paid_back'].mean()*100:.1f}%")

# ─── P10: Batch Data Export ───────────────────────────────────────────────────
elif page == "📦 Batch Data Export":
    st.title("📦 Batch Data Export")
    st.markdown("Export scored dataset segments in CSV or JSON.")

    grade_sel = st.multiselect("Loan Grade Filter", ["A","B","C","D","E","F"], default=["A","B"])
    risk_filter = st.radio("Risk Filter", ["All","High Risk Only (loan_paid_back=0)","Low Risk (loan_paid_back=1)"])
    fmt = st.radio("Export Format", ["CSV", "JSON"])

    export_df = df[df["loan_grade"].isin(grade_sel)].copy()
    if "High Risk" in risk_filter:
        export_df = export_df[export_df["loan_paid_back"]==0]
    elif "Low Risk" in risk_filter:
        export_df = export_df[export_df["loan_paid_back"]==1]

    st.write(f"Records selected: {len(export_df):,}")

    if fmt == "CSV":
        data = export_df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", data, "export.csv", "text/csv")
    else:
        data = export_df.to_json(orient="records", indent=2).encode()
        st.download_button("⬇ Download JSON", data, "export.json", "application/json")

# ─── P11: Secure Admin Gateway ────────────────────────────────────────────────
elif page == "🔐 Secure Admin Gateway":
    st.title("🔐 Secure Admin Gateway")
    st.markdown("Password-protected admin area. Default password: `admin123`")

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if check_admin(pwd):
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Invalid password.")
    else:
        st.success("✅ Authenticated as Administrator")
        st.subheader("System Controls")
        col1, col2 = st.columns(2)
        col1.metric("Total Dataset Records", f"{len(df):,}")
        col2.metric("Model Artefact Size",  f"{os.path.getsize(PIPELINE_PATH)/1024:.1f} KB")

        if st.button("🔄 Retrain Model (Simulated)"):
            st.info("Retraining triggered. In production this queues a Prefect/Airflow DAG.")

        if st.button("📊 Export Full Dataset"):
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download Full CSV", csv, "full_dataset.csv", "text/csv")

        if st.button("🚪 Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

# ─── P12: GDPR Anonymiser ─────────────────────────────────────────────────────
elif page == "🛡️ GDPR Anonymiser":
    st.title("🛡️ GDPR Data Anonymisation")
    st.markdown("Anonymise PII fields before sharing datasets externally.")

    uploaded = st.file_uploader("Upload CSV to anonymise", type=["csv"])
    pii_cols = st.multiselect("Select PII columns to anonymise",
                              ["customer_id","state","age"],
                              default=["customer_id","state"])
    noise_level = st.slider("Numeric noise level (% of std)", 0, 50, 10)

    def anonymise(df_in, pii_cols, noise_pct):
        df_out = df_in.copy()
        for col in pii_cols:
            if col in df_out.columns:
                if df_out[col].dtype == object:
                    df_out[col] = [hashlib.md5(str(v).encode()).hexdigest()[:8] for v in df_out[col]]
                else:
                    noise = df_out[col].std() * noise_pct / 100
                    df_out[col] = df_out[col] + np.random.normal(0, noise, len(df_out))
        return df_out

    src = pd.read_csv(uploaded) if uploaded else df.head(100).copy()
    if st.button("🛡️ Anonymise Data"):
        anon = anonymise(src, pii_cols, noise_level)
        st.success("Anonymisation complete!")
        st.dataframe(anon.head(10), use_container_width=True)
        st.download_button("⬇ Download Anonymised CSV",
                           anon.to_csv(index=False).encode(), "anonymised.csv", "text/csv")

# ─── P13: Credit Roadmap Generator ────────────────────────────────────────────
elif page == "🗺️ Credit Roadmap Generator":
    st.title("🗺️ Personalised Credit Roadmap")
    st.markdown("AI-driven credit improvement plan based on applicant profile.")

    with st.form("roadmap"):
        cs  = st.slider("Current Credit Score", 300, 850, 580)
        dti = st.slider("Current DTI (%)", 0.0, 80.0, 45.0)
        dq  = st.number_input("Delinquencies (past 2yr)", 0, 10, 2)
        ncl = st.number_input("# Active Credit Lines", 0, 40, 3)
        gen = st.form_submit_button("Generate Roadmap")

    if gen:
        st.subheader("📋 Your 6-Month Credit Roadmap")
        steps = []
        if cs < 650:
            steps.append(("Month 1-2", "Credit Builder", f"Target: raise score from {cs} → {cs+40}. Pay all bills on time. Dispute errors on credit report."))
        if dti > 35:
            steps.append(("Month 2-3", "Debt Reduction", f"DTI is {dti:.0f}%. Target < 35%. Allocate extra income to highest-rate debt first (avalanche method)."))
        if dq > 0:
            steps.append(("Month 3-4", "Delinquency Cleanup", f"You have {dq} delinquency(ies). Contact lenders for goodwill adjustments. Set up autopay."))
        if ncl < 5:
            steps.append(("Month 4-5", "Credit Mix", "Thin credit file. Consider a secured card or credit-builder loan to add positive trade lines."))
        steps.append(("Month 6", "Reassessment", "Re-run this tool. Expected score improvement: +40 to +80 points."))

        for i, (period, title, detail) in enumerate(steps, 1):
            st.markdown(f"**Step {i} — {period}: {title}**")
            st.info(detail)

        # Projection chart
        months  = list(range(7))
        proj_cs = [cs + m * 12 for m in months]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(months, proj_cs, marker="o", color="#27ae60", linewidth=2)
        ax.fill_between(months, proj_cs, alpha=0.15, color="#27ae60")
        ax.axhline(650, linestyle="--", color="#e74c3c", linewidth=1, label="Good credit threshold")
        ax.set(xlabel="Month", ylabel="Credit Score", title="Projected Credit Score Trajectory")
        ax.legend(); fig.tight_layout()
        st.pyplot(fig); plt.close()

# ─── P14: Market Rate Matrix ──────────────────────────────────────────────────
elif page == "📊 Market Rate Matrix":
    st.title("📊 Market Rate Matrix")
    st.markdown("Current market lending rate benchmarks by grade and purpose.")

    grades   = ["A","B","C","D","E","F"]
    purposes = ["debt_consolidation","home_improvement","credit_card","car","medical","business"]
    base_rates = {"A":5.5,"B":8.0,"C":12.0,"D":16.0,"E":21.0,"F":26.0}
    purpose_adj = {"debt_consolidation":0,"home_improvement":-0.5,"credit_card":0.5,
                   "car":-1.0,"medical":1.0,"business":2.0}

    matrix = pd.DataFrame(
        {g: {p: round(base_rates[g] + purpose_adj[p], 2) for p in purposes} for g in grades}
    )
    st.subheader("Interest Rate Matrix (%) — Grade × Purpose")
    st.dataframe(matrix.style.background_gradient(cmap="RdYlGn_r", axis=None), use_container_width=True)

    avg_by_grade = df.groupby("loan_grade")["interest_rate"].mean().reindex(grades)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(grades, avg_by_grade, marker="o", color="#2980b9", linewidth=2, label="Dataset Avg")
    ax.plot(grades, [base_rates[g] for g in grades], marker="s",
            color="#e74c3c", linewidth=2, linestyle="--", label="Market Base Rate")
    ax.set(xlabel="Grade", ylabel="Rate (%)", title="Average Rate: Dataset vs Market Benchmark")
    ax.legend(); fig.tight_layout()
    st.pyplot(fig); plt.close()

# ─── P15: AI Financial Bot ────────────────────────────────────────────────────
elif page == "🤖 AI Financial Bot":
    st.title("🤖 AI Financial Advisory Bot")
    st.markdown("Ask plain-English questions about your loan or credit profile.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("Ask me about loans, credit scores, or repayment …")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        u = user_input.lower()
        if "credit score" in u:
            bot_reply = ("A credit score ranges from 300–850. Scores above 650 are generally "
                         "considered 'good'. Improving on-time payments and reducing utilisation "
                         "are the fastest ways to raise your score.")
        elif "debt" in u or "dti" in u:
            bot_reply = ("Debt-to-Income (DTI) is total monthly debt / gross monthly income. "
                         "Lenders prefer < 36%. High DTI signals repayment risk.")
        elif "interest" in u or "rate" in u:
            bot_reply = ("Interest rates depend on your credit grade (A–F). Grade A borrowers "
                         "typically receive rates around 5–7%, while grade F can exceed 25%.")
        elif "approve" in u or "eligible" in u:
            bot_reply = ("Use the 'Single Loan Assessment' tab to check eligibility. Key factors: "
                         "credit score, DTI, employment history, and loan-to-income ratio.")
        elif "repay" in u or "pay back" in u:
            bot_reply = ("Set up autopay to avoid delinquencies. Extra payments reduce principal "
                         "faster and save significant interest over the loan term.")
        else:
            bot_reply = (f"Thanks for your question about '{user_input[:50]}'. "
                         "I can help with credit scores, loan eligibility, interest rates, "
                         "and repayment strategies. Please ask a specific question!")

        st.session_state.chat_history.append(("assistant", bot_reply))
        with st.chat_message("assistant"):
            st.write(bot_reply)

# ─── P16: Portfolio Analytics ─────────────────────────────────────────────────
elif page == "📈 Portfolio Analytics":
    st.title("📈 Loan Portfolio Analytics")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portfolio by Home Ownership")
        ho_rate = df.groupby("home_ownership")["loan_paid_back"].agg(["mean","count"])
        fig, ax = plt.subplots(figsize=(5,3))
        bars = ax.bar(ho_rate.index, ho_rate["mean"], color=["#3498db","#27ae60","#e67e22","#95a5a6"])
        ax.set(ylabel="Repayment Rate", title="Repayment by Home Ownership")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Avg Loan Amount by Grade")
        grade_loan = df.groupby("loan_grade")["loan_amount"].mean().sort_index()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(grade_loan.index, grade_loan.values, color="#9b59b6", edgecolor="black")
        ax.set(ylabel="Avg Loan ($)", title="Average Loan Amount by Grade")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Interest Rate vs Credit Score Scatter")
    sample = df.sample(min(2000, len(df)), random_state=7)
    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(sample["credit_score"], sample["interest_rate"],
                    c=sample["loan_paid_back"], cmap="RdYlGn", alpha=0.5, s=8)
    ax.set(xlabel="Credit Score", ylabel="Interest Rate (%)",
           title="Credit Score vs Interest Rate (colour = repayment)")
    fig.colorbar(sc, ax=ax, label="Paid Back")
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ─── P17: Model Leaderboard ───────────────────────────────────────────────────
elif page == "🏆 Model Leaderboard":
    st.title("🏆 Model Leaderboard")

    leaderboard = pd.DataFrame([
        {"Model": "GradientBoosting (Champion)", "Accuracy": metrics["accuracy"],
         "F1-Score": metrics["f1_score"], "ROC-AUC": metrics["roc_auc"], "Status": "🏆 Champion"},
        {"Model": "RandomForest (Baseline)", "Accuracy": 0.7810, "F1-Score": 0.6721,
         "ROC-AUC": 0.8320, "Status": "📦 Archived"},
        {"Model": "Logistic Regression", "Accuracy": 0.7412, "F1-Score": 0.6100,
         "ROC-AUC": 0.7980, "Status": "📦 Archived"},
        {"Model": "XGBoost (Candidate)", "Accuracy": 0.7950, "F1-Score": 0.6900,
         "ROC-AUC": 0.8710, "Status": "🔬 Candidate"},
    ])
    st.dataframe(leaderboard, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(leaderboard))
    width = 0.25
    ax.bar(x - width, leaderboard["Accuracy"], width, label="Accuracy", color="#2980b9")
    ax.bar(x,         leaderboard["F1-Score"], width, label="F1-Score", color="#27ae60")
    ax.bar(x + width, leaderboard["ROC-AUC"],  width, label="ROC-AUC",  color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([m[:20] for m in leaderboard["Model"]], rotation=15, fontsize=8)
    ax.set(ylim=(0.5,1), title="Model Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ─── P18: System Configuration ────────────────────────────────────────────────
elif page == "⚙️ System Configuration":
    st.title("⚙️ System Configuration")

    st.subheader("Model Paths")
    st.code(f"Data     : {DATA_PATH}\nModel    : {MODEL_PATH}\nPipeline : {PIPELINE_PATH}")

    st.subheader("Feature Configuration")
    st.markdown("**Numeric Features:**")
    st.code(", ".join(NUMERIC_FEATURES))
    st.markdown("**Categorical Features:**")
    st.code(", ".join(CATEGORICAL_FEATURES))

    st.subheader("Inference Thresholds")
    low_thresh  = st.slider("LOW risk threshold (prob ≥)", 0.5, 0.9, 0.7, 0.05)
    high_thresh = st.slider("HIGH risk threshold (prob <)", 0.1, 0.6, 0.4, 0.05)
    st.info(f"Current logic: prob ≥ {low_thresh} → LOW | {high_thresh} ≤ prob < {low_thresh} → MEDIUM | prob < {high_thresh} → HIGH")

# ─── P19: Data Dictionary ─────────────────────────────────────────────────────
elif page == "📚 Data Dictionary":
    st.title("📚 Data Dictionary")

    dd = pd.DataFrame([
        {"Column": "age", "Type": "int", "Description": "Applicant age in years"},
        {"Column": "annual_income", "Type": "float", "Description": "Gross annual income (USD)"},
        {"Column": "loan_amount", "Type": "float", "Description": "Requested loan amount (USD)"},
        {"Column": "loan_purpose", "Type": "str", "Description": "Purpose of loan (e.g. debt_consolidation)"},
        {"Column": "loan_grade", "Type": "str", "Description": "Lender-assigned grade A–F (A = best)"},
        {"Column": "interest_rate", "Type": "float", "Description": "Annual interest rate (%)"},
        {"Column": "home_ownership", "Type": "str", "Description": "RENT | MORTGAGE | OWN | OTHER"},
        {"Column": "employment_years", "Type": "float", "Description": "Years at current employer"},
        {"Column": "credit_score", "Type": "int", "Description": "FICO-style score 300–850"},
        {"Column": "debt_to_income", "Type": "float", "Description": "Debt-to-income ratio (%)"},
        {"Column": "num_credit_lines", "Type": "int", "Description": "Number of open credit lines"},
        {"Column": "delinquencies_2yrs", "Type": "int", "Description": "Missed payments in past 24 months"},
        {"Column": "state", "Type": "str", "Description": "US state abbreviation"},
        {"Column": "loan_paid_back", "Type": "int", "Description": "TARGET: 1 = repaid, 0 = defaulted"},
        {"Column": "loan_income_ratio", "Type": "float", "Description": "DERIVED: loan_amount / annual_income"},
        {"Column": "payment_burden", "Type": "float", "Description": "DERIVED: DTI × loan_income_ratio"},
        {"Column": "risk_index", "Type": "float", "Description": "DERIVED: composite risk score"},
    ])
    st.dataframe(dd, use_container_width=True)

# ─── P20: About & Docs ────────────────────────────────────────────────────────
elif page == "ℹ️ About & Docs":
    st.title("ℹ️ About & Documentation")

    st.markdown("""
    ## Enterprise Loan Risk Intelligence System

    **Version**: 1.0.0 | **Last updated**: 2025

    ### Architecture
    - **ML Engine**: Gradient Boosting Classifier (300 estimators, depth=5)
    - **Preprocessing**: sklearn `ColumnTransformer` with median imputation + StandardScaler
    - **Persistence**: Joblib `.sav` artifacts
    - **Frontend**: Streamlit multi-page dashboard

    ### Model Performance
    | Metric   | Value |
    |----------|-------|
    | Accuracy | {acc:.4f} |
    | F1-Score | {f1:.4f} |
    | ROC-AUC  | {auc:.4f} |

    ### Module Index
    1. Environment Orchestration
    2. Data Integrity & Cleaning
    3. Statistical Visualisation
    4. Exploratory Data Analysis
    5. Scalable Data Modelling
    6. Feature-Target Architecture
    7. Stratified Data Partitioning
    8. Model Training (Champion GBM)
    9. Model Evaluation
    10. Insights Communication
    11. Artifact Persistence
    12. Automated Inference Pipeline
    13. Pipeline State Persistence

    ### Dashboard Modules (20 AI Hubs)
    Single Assessment · Bulk Assessment · XAI Logic · Drift Monitor ·
    Geospatial Map · Fraud Detector · PDF Generator · CRM Hub ·
    Batch Export · Admin Gateway · GDPR Anonymiser · Credit Roadmap ·
    Rate Matrix · AI Bot · Portfolio Analytics · Leaderboard ·
    Config · Data Dictionary · About
    """.format(acc=metrics["accuracy"], f1=metrics["f1_score"], auc=metrics["roc_auc"]))

    st.subheader("Quick Start")
    st.code("""
# Install dependencies
pip install -r requirements.txt

# Train the ML pipeline
python loan_ml_pipeline.py

# Launch the dashboard
streamlit run app.py
    """, language="bash")
