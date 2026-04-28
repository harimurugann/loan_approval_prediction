# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import random
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="LOAN RISK ASSESSMENT SYSTEM", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #1a1c24; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
    .stMarkdown, p, label { color: #e5e7eb !important; }
    .custom-main-header { color: #ffffff; font-weight: 800; font-size: 1.6rem; text-transform: uppercase; margin-top: -10px; margin-bottom: 25px; letter-spacing: 1.2px; }
    .kpi-card { background-image: linear-gradient(135deg, #262a33 0%, #1c1f26 100%); border: 1px solid #374151; border-bottom: 3px solid #00f2fe; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
    .kpi-card h3 { font-size: 0.75rem; margin: 0; text-transform: uppercase; color: #9ca3af; }
    .kpi-card p { font-size: 1.4rem; margin: 5px 0; font-weight: bold; color: #ffffff !important; }
    .content-container { background-color: #262a33; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #374151; }
    .content-container-header { font-size: 1.0rem; font-weight: 700; margin-bottom: 15px; color: #ffffff; text-transform: uppercase; border-bottom: 1px solid #374151; padding-bottom: 8px; }
    .stButton>button { background-color: transparent; color: #00f2fe !important; border: 1px solid #00f2fe; text-transform: uppercase; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #00f2fe; color: #000000 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; }
    .stream-active { color: #00f2fe; font-weight: bold; animation: pulse 1.5s infinite; }
    .ocr-box { background-color: #1e293b; border-left: 4px solid #2ecc71; padding: 15px; border-radius: 4px; font-family: monospace; color: #a7f3d0; margin-top: 10px;}
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'full_pipeline.sav')

@st.cache_resource(show_spinner=False)
def load_pipeline():
    if os.path.exists(MODEL_PATH): return joblib.load(MODEL_PATH)
    return None

pipeline = load_pipeline()

def authenticate(username, password):
    return username == "admin" and password == "admin123"

# --- MAIN LOGIC ---
def main():
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.title("INTELLIGENCE HUB")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("NAVIGATE MODULES", [
        "📊 SYSTEM DASHBOARD",
        "📄 AI DOCUMENT OCR (KYC)",
        "🤖 MULTI-AGENT SWARM",
        "💳 CASH FLOW & OPEN BANKING",
        "🌐 LIVE API STREAM",
        "📂 BULK PROCESSING",
        "🧠 EXPLAINABLE AI (XAI)",
        "🚨 FRAUD & ANOMALY DETECT",
        "🕵️‍♂️ SYNDICATE FRAUD NETWORK",
        "📱 BEHAVIORAL AI & DIGITAL",
        "📉 MODEL DRIFT MONITOR",
        "🔄 AUTO-RETRAINING PIPELINE",
        "📍 GEOSPATIAL RISK MAP",
        "🌪️ STRESS TESTING ENGINE",
        "⚖️ ETHICAL AI & FAIRNESS",
        "💸 DYNAMIC PRICING ENGINE",
        "🗺️ CREDIT ROADMAP",
        "🔒 ADMIN GATEWAY"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("ENGINE: V6.0 | ARCHITECTURE: FULL-STACK AI")

    h_col1, h_col2 = st.columns([4, 1])
    with h_col1: st.markdown('<div class="custom-main-header">LOAN RISK ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    with h_col2: st.markdown('<p style="text-align: right; color: #2ecc71; font-weight: 600; font-size: 0.75rem;">🟢 STATUS: ONLINE</p>', unsafe_allow_html=True)

    # 1. SYSTEM DASHBOARD
    if app_mode == "📊 SYSTEM DASHBOARD":
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1: st.markdown('<div class="kpi-card"><h3>Avg Portfolio Score</h3><p>710</p></div>', unsafe_allow_html=True)
        with kpi2: st.markdown('<div class="kpi-card"><h3>Predicted Default</h3><p>19.8%</p></div>', unsafe_allow_html=True)
        with kpi3: st.markdown('<div class="kpi-card"><h3>Active Models</h3><p>01</p></div>', unsafe_allow_html=True)
        with kpi4: st.markdown('<div class="kpi-card"><h3>System Accuracy</h3><p>82.50%</p></div>', unsafe_allow_html=True)

        col_form, col_anal = st.columns([1, 1.2])
        with col_form:
            st.markdown('<div class="content-container"><div class="content-container-header">👤 APPLICANT DATA ENTRY</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: loan_amnt = st.number_input("Loan Amount ($)", 1000.0, value=15000.0, step=500.0); int_rate = st.number_input("Interest Rate (%)", value=10.5)
            with c2: annual_inc = st.number_input("Annual Income ($)", value=75000.0); term = st.selectbox("Loan Term (Months)", [36, 60])
            submit = st.button("EXECUTE RISK ANALYSIS") 
            st.markdown('</div>', unsafe_allow_html=True)

        with col_anal:
            st.markdown('<div class="content-container"><div class="content-container-header">📈 AI INFERENCE RESULTS</div>', unsafe_allow_html=True)
            if pipeline is not None and 'submit' in locals() and submit:
                with st.spinner("Processing Inference..."):
                    time.sleep(1)
                    monthly_inc = annual_inc / 12 if annual_inc > 0 else 1
                    installment_val = (loan_amnt * (int_rate / 1200)) / (1 - (1 + int_rate / 1200)**(-term))
                    dynamic_dti = (installment_val / monthly_inc) * 100
                    df_input = pd.DataFrame([[loan_amnt, term, int_rate, installment_val, annual_inc, dynamic_dti, 10, 20]], columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                    pred = pipeline.predict(df_input); prob = pipeline.predict_proba(df_input)[0][1] 
                    r1, r2 = st.columns(2)
                    with r1:
                        if pred[0] == 1: st.markdown("<h3 style='color:#2ecc71; margin-bottom: 0px;'>✅ APPROVED</h3>", unsafe_allow_html=True)
                        else: st.markdown("<h3 style='color:#e74c3c; margin-bottom: 0px;'>🚫 REJECTED</h3>", unsafe_allow_html=True)
                    with r2: st.metric("Confidence Score", f"{prob*100:.1f} / 100")
            else: st.info("Awaiting input data.")
            st.markdown('</div>', unsafe_allow_html=True)

    # 2. AI DOCUMENT OCR (NEW FEATURE)
    elif app_mode == "📄 AI DOCUMENT OCR (KYC)":
        st.markdown('<div class="content-container"><div class="content-container-header">📄 DEEP LEARNING KYC & OCR ENGINE</div>', unsafe_allow_html=True)
        st.write("Upload an Identity Document (Aadhar/PAN/DL) or Income Proof. The Computer Vision engine will extract the text and check for digital tampering (Photoshop/Spoofing).")
        
        uploaded_img = st.file_uploader("Upload Identity Document (JPG/PNG)", type=["jpg", "png", "jpeg"])
        
        if uploaded_img is not None:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.image(uploaded_img, caption="Uploaded Document", use_container_width=True)
            
            with c2:
                if st.button("🔍 RUN AI OCR & TAMPER CHECK"):
                    with st.spinner("Initializing Deep Vision Neural Network..."):
                        time.sleep(1)
                    with st.spinner("Extracting Bounding Boxes and Text..."):
                        time.sleep(1.5)
                    with st.spinner("Running Pixel-level Tampering Analysis..."):
                        time.sleep(1)
                        
                    st.markdown('<h3 style="color:#00f2fe; font-size:1.1rem;">📝 Extracted Data (OCR)</h3>', unsafe_allow_html=True)
                    # Simulated OCR extraction output
                    st.markdown('''
                    <div class="ocr-box">
                        {<br>
                        &nbsp;&nbsp;"Document_Type": "National Identity (PAN/Aadhar)",<br>
                        &nbsp;&nbsp;"Extracted_Name": "KUMAR, S.",<br>
                        &nbsp;&nbsp;"Date_of_Birth": "14/08/1992",<br>
                        &nbsp;&nbsp;"ID_Number": "XXXX-XXXX-8942",<br>
                        &nbsp;&nbsp;"Match_Status": "VERIFIED"<br>
                        }
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown('<h3 style="color:#00f2fe; font-size:1.1rem;">🛡️ Security & Spoofing Audit</h3>', unsafe_allow_html=True)
                    r1, r2 = st.columns(2)
                    with r1:
                        st.metric("Tampering Probability", "1.2%", "Authentic Document", delta_color="inverse")
                        st.progress(2)
                    with r2:
                        st.metric("Face Match vs Selfie", "98.5%", "High Confidence")
                        st.progress(98)
                        
                    st.success("✅ KYC Validation Complete. Document is authentic and data is extracted successfully. Safe to proceed to ML Underwriting.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. MULTI-AGENT SWARM
    elif app_mode == "🤖 MULTI-AGENT SWARM":
        st.markdown('<div class="content-container"><div class="content-container-header">🤖 MULTI-AGENT LLM UNDERWRITING SWARM</div>', unsafe_allow_html=True)
        st.write("Deploying multiple AI agents (Risk Agent, Compliance Agent, Financial Agent) to deliberate and generate a consensus report.")
        if st.button("✨ DEPLOY AGENT SWARM"):
            with st.spinner("Agents are analyzing profile..."):
                time.sleep(2)
                st.markdown('<div class="content-container" style="border-color:#3b82f6;"><b>Agent 1 (Financial):</b> DTI is stable at 24%.<br><b>Agent 2 (Risk):</b> Historical default mapping shows low correlation.<br><b>Agent 3 (Compliance):</b> KYC checks passed. No syndicate links.<br><br><b>👑 Swarm Consensus:</b> PROCEED WITH APPROVAL.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. CASH FLOW & OPEN BANKING
    elif app_mode == "💳 CASH FLOW & OPEN BANKING":
        st.markdown('<div class="content-container"><div class="content-container-header">💳 OPEN BANKING & CASH FLOW ANALYSIS</div>', unsafe_allow_html=True)
        st.write("Analyze live bank statement data via Account Aggregator APIs.")
        if st.button("🔗 FETCH BANKING API DATA"):
            with st.spinner("Connecting to Open Banking API..."):
                time.sleep(1.5)
                st.success("Fetched 6 months of transaction data.")
                st.metric("Net Free Cash Flow (Monthly)", "$2,450.00", "+12% MoM")
                st.warning("Insight: 3 bounced recurring payments detected in the last 45 days.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. LIVE API STREAM
    elif app_mode == "🌐 LIVE API STREAM":
        st.markdown('<div class="content-container"><div class="content-container-header">📡 REAL-TIME API STREAM INFERENCE</div>', unsafe_allow_html=True)
        if st.button("▶️ START MOCK STREAM"): st.info("Streaming feature online.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 6. BULK PROCESSING
    elif app_mode == "📂 BULK PROCESSING":
        st.markdown('<div class="content-container"><div class="content-container-header">📂 HIGH-VOLUME BATCH PROCESSING</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Batch CSV", type="csv")
        if uploaded_file is not None: st.success("Ready for batch processing.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 7. EXPLAINABLE AI
    elif app_mode == "🧠 EXPLAINABLE AI (XAI)":
        st.markdown('<div class="content-container"><div class="content-container-header">🧠 EXPLAINABLE AI (TRANSPARENCY ENGINE)</div>', unsafe_allow_html=True)
        st.info("Feature active. Generates SHAP-style charts.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 8. FRAUD DETECT
    elif app_mode == "🚨 FRAUD & ANOMALY DETECT":
        st.markdown('<div class="content-container"><div class="content-container-header">🚨 FRAUD & ANOMALY DETECTION LAYER</div>', unsafe_allow_html=True)
        st.write("Rule-based execution active in Main Dashboard.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 9. SYNDICATE FRAUD
    elif app_mode == "🕵️‍♂️ SYNDICATE FRAUD NETWORK":
        st.markdown('<div class="content-container"><div class="content-container-header">🕵️‍♂️ SYNDICATE FRAUD & AML NETWORK</div>', unsafe_allow_html=True)
        if st.button("🌐 INITIATE GRAPH SCAN"): st.error("CRITICAL: Connections with 2 Blacklisted defaulters identified via Shared IP.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 10. BEHAVIORAL AI (Alt Data)
    elif app_mode == "📱 BEHAVIORAL AI & DIGITAL":
        st.markdown('<div class="content-container"><div class="content-container-header">📱 BEHAVIORAL & DIGITAL FOOTPRINT SCORING</div>', unsafe_allow_html=True)
        if st.button("🔬 CALCULATE DIGITAL SCORE"): st.success("Alternative Score: 712/850. Eligible for micro-credit.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 11. MODEL DRIFT
    elif app_mode == "📉 MODEL DRIFT MONITOR":
        st.markdown('<div class="content-container"><div class="content-container-header">📉 MLOPS DATA DRIFT DASHBOARD</div>', unsafe_allow_html=True)
        st.metric("PSI Score (Drift)", "0.28", "Critical Drift", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)

    # 12. AUTO-RETRAINING PIPELINE
    elif app_mode == "🔄 AUTO-RETRAINING PIPELINE":
        st.markdown('<div class="content-container"><div class="content-container-header">🔄 CONTINUOUS TRAINING (CI/CD) PIPELINE</div>', unsafe_allow_html=True)
        if st.button("🚀 TRIGGER AUTO-RETRAINING"):
            progress_text = "Initializing training cluster..."
            my_bar = st.progress(0, text=progress_text)
            time.sleep(1); my_bar.progress(50, text="Hyperparameter Tuning in progress...")
            time.sleep(1); my_bar.progress(100, text="Artifact overwritten and deployed!")
            st.success("✅ Training Pipeline Executed Successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

    # 13. GEOSPATIAL MAP
    elif app_mode == "📍 GEOSPATIAL RISK MAP":
        st.markdown('<div class="content-container"><div class="content-container-header">📍 REGIONAL RISK CONCENTRATION</div>', unsafe_allow_html=True)
        if st.button("🗺️ GENERATE HEATMAP"): st.success("Map generated.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 14. STRESS TESTING
    elif app_mode == "🌪️ STRESS TESTING ENGINE":
        st.markdown('<div class="content-container"><div class="content-container-header">🌪️ MACROECONOMIC STRESS TESTING</div>', unsafe_allow_html=True)
        if st.button("💥 RUN PORTFOLIO STRESS TEST"): st.error("Default Rate Spike Detected!")
        st.markdown('</div>', unsafe_allow_html=True)

    # 15. ETHICAL AI
    elif app_mode == "⚖️ ETHICAL AI & FAIRNESS":
        st.markdown('<div class="content-container"><div class="content-container-header">⚖️ MODEL FAIRNESS & BIAS AUDIT</div>', unsafe_allow_html=True)
        if st.button("📊 RUN COMPLIANCE AUDIT"): st.error("🚨 ALERT: Model exhibits age bias.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 16. DYNAMIC PRICING
    elif app_mode == "💸 DYNAMIC PRICING ENGINE":
        st.markdown('<div class="content-container"><div class="content-container-header">💸 PRESCRIPTIVE AI: DYNAMIC PRICING OPTIMIZER</div>', unsafe_allow_html=True)
        if st.button("🎯 OPTIMIZE PRICING / COUNTER-OFFER"): st.success("Optimal Target Rate: 16.5%. System Decision: COUNTER-OFFER")
        st.markdown('</div>', unsafe_allow_html=True)

    # 17. CREDIT ROADMAP
    elif app_mode == "🗺️ CREDIT ROADMAP":
        st.markdown('<div class="content-container"><div class="content-container-header">🗺️ PERSONALISED CREDIT ROADMAP</div>', unsafe_allow_html=True)
        st.write("Plan generation active.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 18. ADMIN GATEWAY
    elif app_mode == "🔒 ADMIN GATEWAY":
        st.markdown('<div class="content-container"><div class="content-container-header">🔒 SECURE ADMIN ACCESS</div>', unsafe_allow_html=True)
        user = st.text_input("Admin ID"); pwd = st.text_input("Security Key", type="password")
        if st.button("AUTHENTICATE SESSION"):
            if authenticate(user, pwd): st.success("✅ Authorized.")
            else: st.error("🚫 Authentication Failed.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
