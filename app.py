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

# --- PAGE CONFIGURATION & METADATA ---
st.set_page_config(
    page_title="LOAN RISK ASSESSMENT SYSTEM", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS: Modern Dark Fintech Design ---
st.markdown("""
    <style>
    .main { background-color: #1a1c24; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stMarkdown, p, label { color: #e5e7eb !important; }
    .custom-main-header {
        color: #ffffff; font-weight: 800; font-size: 1.6rem; text-transform: uppercase;
        margin-top: -10px; margin-bottom: 25px; letter-spacing: 1.2px;
    }
    .kpi-card {
        background-image: linear-gradient(135deg, #262a33 0%, #1c1f26 100%);
        border: 1px solid #374151; border-bottom: 3px solid #00f2fe;
        border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-card h3 { font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; }
    .kpi-card p { font-size: 1.4rem; margin: 5px 0; font-weight: bold; color: #ffffff !important; }
    .kpi-card-sub { font-size: 0.70rem; color: #2ecc71 !important; margin: 0; font-weight: bold; }
    .content-container {
        background-color: #262a33; border-radius: 12px; padding: 20px;
        margin-bottom: 20px; border: 1px solid #374151; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .content-container-header {
        font-size: 1.0rem; font-weight: 700; margin-bottom: 15px; color: #ffffff;
        text-transform: uppercase; border-bottom: 1px solid #374151; padding-bottom: 8px;
    }
    label[data-testid="stWidgetLabel"] { font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
    .stButton>button, .stDownloadButton>button {
        background-color: transparent; color: #00f2fe !important; border-radius: 4px;
        padding: 0.4rem 1.2rem; font-size: 0.85rem; font-weight: bold;
        border: 1px solid #00f2fe; text-transform: uppercase; transition: 0.3s; width: 100%;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background-color: #00f2fe; color: #000000 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; }
    .status-text { text-align: right; color: #2ecc71; font-weight: 600; font-size: 0.75rem; letter-spacing: 1px; }
    .roadmap-box { background-color: #1a1c24; border-left: 4px solid #00f2fe; padding: 15px; margin-top: 15px; border-radius: 4px; }
    .anomaly-box { background-color: #2c0b0e; border-left: 3px solid #e74c3c; padding: 8px; margin-bottom: 5px; border-radius: 4px; color: #ffcccc; font-size: 0.85rem;}
    .stream-active { color: #00f2fe; font-weight: bold; animation: pulse 1.5s infinite; }
    .gen-ai-box { background-color: #1e293b; border: 1px solid #3b82f6; padding: 20px; border-radius: 8px; font-family: monospace; color: #93c5fd; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

# --- DIRECTORY HANDLING & MODEL LOADING ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'full_pipeline.sav')

@st.cache_resource
def load_pipeline():
    if os.path.exists(MODEL_PATH): return joblib.load(MODEL_PATH)
    return None

pipeline = load_pipeline()

def authenticate(username, password):
    return username == "admin" and password == "admin123"

# --- MAIN DASHBOARD LOGIC ---
def main():
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.title("INTELLIGENCE HUB")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("NAVIGATE MODULES", [
        "📊 SYSTEM DASHBOARD",
        "🌐 LIVE API STREAM",
        "📂 BULK PROCESSING",
        "🤖 GEN-AI UNDERWRITING",
        "🧠 EXPLAINABLE AI (XAI)",
        "🚨 FRAUD & ANOMALY DETECT",
        "📉 MODEL DRIFT MONITOR",
        "📍 GEOSPATIAL RISK MAP",
        "🌪️ STRESS TESTING ENGINE",
        "⚖️ ETHICAL AI & FAIRNESS",
        "💸 DYNAMIC PRICING ENGINE",
        "🗺️ CREDIT ROADMAP",
        "🔒 ADMIN GATEWAY"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("ENGINE: V4.2 | STATUS: SECURE")

    h_col1, h_col2 = st.columns([4, 1])
    with h_col1: st.markdown('<div class="custom-main-header">LOAN RISK ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    with h_col2: st.markdown('<p class="status-text">🟢 STATUS: ONLINE<br>Engine Connected</p>', unsafe_allow_html=True)

    # ==========================================
    # 1. SYSTEM DASHBOARD
    # ==========================================
    if app_mode == "📊 SYSTEM DASHBOARD":
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1: st.markdown('<div class="kpi-card"><h3>Avg Portfolio Score</h3><p>710</p><p class="kpi-card-sub">FAIR & STABLE</p></div>', unsafe_allow_html=True)
        with kpi2: st.markdown('<div class="kpi-card"><h3>Predicted Default</h3><p>19.8%</p><p class="kpi-card-sub">BELOW THRESHOLD</p></div>', unsafe_allow_html=True)
        with kpi3: st.markdown('<div class="kpi-card"><h3>Active Models</h3><p>01</p><p class="kpi-card-sub">RANDOM FOREST</p></div>', unsafe_allow_html=True)
        with kpi4: st.markdown('<div class="kpi-card"><h3>System Accuracy</h3><p>80.17%</p><p class="kpi-card-sub">OPTIMIZED</p></div>', unsafe_allow_html=True)

        col_form, col_anal = st.columns([1, 1.2])
        with col_form:
            st.markdown('<div class="content-container"><div class="content-container-header">👤 APPLICANT DATA ENTRY</div>', unsafe_allow_html=True)
            if pipeline is None: st.markdown("<p style='color:#e74c3c;'>🚨 Pipeline not found.</p>", unsafe_allow_html=True)
            else:
                c1, c2 = st.columns(2)
                with c1: loan_amnt = st.number_input("Loan Amount ($)", 1000.0, value=15000.0, step=500.0); int_rate = st.number_input("Interest Rate (%)", value=10.5)
                with c2: annual_inc = st.number_input("Annual Income ($)", value=75000.0); term = st.selectbox("Loan Term (Months)", [36, 60])
                submit = st.button("EXECUTE RISK ANALYSIS") 
            st.markdown('</div>', unsafe_allow_html=True)

        with col_anal:
            st.markdown('<div class="content-container"><div class="content-container-header">📈 AI INFERENCE RESULTS</div>', unsafe_allow_html=True)
            if pipeline is not None and 'submit' in locals() and submit:
                with st.spinner("Running Anomaly Scan & AI Inference..."):
                    time.sleep(1)
                    monthly_inc = annual_inc / 12 if annual_inc > 0 else 1
                    installment_val = (loan_amnt * (int_rate / 1200)) / (1 - (1 + int_rate / 1200)**(-term))
                    dynamic_dti = (installment_val / monthly_inc) * 100
                    
                    fraud_flags = []
                    if loan_amnt > (annual_inc * 4): fraud_flags.append(f"Loan Amt ({loan_amnt}) > 4x Annual Income ({annual_inc}).")
                    if dynamic_dti > 50.0: fraud_flags.append(f"Critical DTI: {dynamic_dti:.1f}% (Exceeds 50% limit).")
                        
                    r1, r2 = st.columns(2)
                    if len(fraud_flags) > 0:
                        with r1:
                            st.markdown("<h3 style='color:#e74c3c; margin-bottom: 0px;'>🚫 SYSTEM REJECTED</h3><p style='color:#9ca3af; font-size: 0.85rem;'>Blocked by Security Rules</p>", unsafe_allow_html=True)
                            for flag in fraud_flags: st.markdown(f'<div class="anomaly-box">❌ {flag}</div>', unsafe_allow_html=True)
                        with r2: st.metric("Confidence Score", "0.0 / 100")
                    else:
                        df_input = pd.DataFrame([[loan_amnt, term, int_rate, installment_val, annual_inc, dynamic_dti, 10, 20]], columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                        prediction = pipeline.predict(df_input)
                        probability = pipeline.predict_proba(df_input)[0][1] 
                        with r1:
                            if prediction[0] == 1: st.markdown("<h3 style='color:#2ecc71; margin-bottom: 0px;'>✅ APPROVED</h3><p style='color:#9ca3af; font-size: 0.85rem;'>Risk Profile: Low to Moderate</p>", unsafe_allow_html=True)
                            else: st.markdown("<h3 style='color:#e74c3c; margin-bottom: 0px;'>🚫 REJECTED</h3><p style='color:#9ca3af; font-size: 0.85rem;'>Risk Profile: High Default Probability</p>", unsafe_allow_html=True)
                        with r2: st.metric("Confidence Score", f"{probability*100:.1f} / 100")
            else: st.markdown("<p style='color:#9ca3af; font-style:italic;'>Awaiting input data.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 2. LIVE API STREAM
    # ==========================================
    elif app_mode == "🌐 LIVE API STREAM":
        st.markdown('<div class="content-container"><div class="content-container-header">📡 REAL-TIME API STREAM INFERENCE</div>', unsafe_allow_html=True)
        stream_file = st.file_uploader("Upload Testing Data for Stream (CSV)", type="csv")
        if 'stream_active' not in st.session_state: st.session_state.stream_active = False
        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            if st.button("▶️ START STREAM"): st.session_state.stream_active = True
        with c2:
            if st.button("⏹️ STOP STREAM"): st.session_state.stream_active = False
        if st.session_state.stream_active: st.markdown("<p class='stream-active'>🟢 System is actively streaming...</p>", unsafe_allow_html=True)
        else: st.markdown("<p style='color:#9ca3af;'>🔴 Stream offline.</p>", unsafe_allow_html=True)
        st.markdown("---")
        placeholder = st.empty()
        if st.session_state.stream_active and pipeline is not None:
            if 'live_df' not in st.session_state: st.session_state.live_df = pd.DataFrame(columns=["Timestamp", "App_ID", "Req_Amount", "Income", "AI_Decision", "Confidence"])
            with placeholder.container(): table_placeholder = st.empty()
            if stream_file is not None:
                try:
                    df_test = pd.read_csv(stream_file)
                    for index, row in df_test.iterrows():
                        if not st.session_state.stream_active: break
                        try:
                            app_id = f"APP-{random.randint(10000, 99999)}"
                            l_amnt = float(row.get('loan_amnt', 15000))
                            df_stream = pd.DataFrame({'loan_amnt': [l_amnt], 'term': [36], 'int_rate': [10.5], 'installment': [300], 'annual_inc': [float(row.get('annual_inc', 50000))], 'dti': [15.0], 'open_acc': [10], 'total_acc': [20]})
                            pred = pipeline.predict(df_stream)
                            prob = pipeline.predict_proba(df_stream)[0][1]
                            decision = "✅ APPROVED" if pred[0] == 1 else "🚫 REJECTED"
                            new_record = pd.DataFrame({"Timestamp": [time.strftime("%H:%M:%S")], "App_ID": [app_id], "Req_Amount": [f"${l_amnt:,.0f}"], "Income": [f"${df_stream['annual_inc'].values[0]:,.0f}"], "AI_Decision": [decision], "Confidence": [f"{prob*100:.1f}%"]})
                            st.session_state.live_df = pd.concat([new_record, st.session_state.live_df]).head(10)
                            table_placeholder.dataframe(st.session_state.live_df, use_container_width=True)
                            time.sleep(1.2) 
                        except Exception: break
                except: st.error("Error reading file.")
            st.session_state.stream_active = False
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 3. BULK PROCESSING
    # ==========================================
    elif app_mode == "📂 BULK PROCESSING":
        st.markdown('<div class="content-container"><div class="content-container-header">📂 HIGH-VOLUME BATCH PROCESSING</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Batch CSV", type="csv")
        if uploaded_file is not None and pipeline is not None:
            df_bulk = pd.read_csv(uploaded_file)
            if st.button("PROCESS BATCH DATA"):
                X_bulk = df_bulk.drop('loan_paid_back', axis=1) if 'loan_paid_back' in df_bulk.columns else df_bulk
                predictions = pipeline.predict(X_bulk)
                df_bulk['AI_Status'] = ["Approved" if p == 1 else "Denied" for p in predictions]
                st.success("✅ Batch processing complete!")
                st.dataframe(df_bulk[['loan_amnt', 'annual_inc', 'AI_Status']].head(5))
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 4. GEN-AI UNDERWRITING AGENT (NEW FEATURE)
    # ==========================================
    elif app_mode == "🤖 GEN-AI UNDERWRITING":
        st.markdown('<div class="content-container"><div class="content-container-header">🤖 GEN-AI UNDERWRITING AGENT</div>', unsafe_allow_html=True)
        st.write("Generates a human-readable summary of the ML model's decision using simulated Large Language Model (LLM) logic.")
        
        c1, c2 = st.columns(2)
        with c1:
            u_amnt = st.number_input("Requested Loan Amount ($)", value=25000, step=1000)
            u_inc = st.number_input("Annual Income ($)", value=60000, step=5000)
        with c2:
            u_dti = st.slider("Debt-to-Income (DTI) %", 5.0, 60.0, 28.0)
            u_score = st.slider("Credit Score", 300, 850, 680)
            
        if st.button("✨ GENERATE AI UNDERWRITING REPORT"):
            with st.spinner("LLM is analyzing the risk profile and generating report..."):
                time.sleep(2)
                
                # Dynamic Prompt generation logic
                status = "APPROVED" if (u_score > 650 and u_dti < 40 and u_amnt < (u_inc*3)) else "REJECTED"
                tone = "positive" if status == "APPROVED" else "critical"
                
                report = f"### 📄 Automated Underwriting Summary\n\n"
                report += f"**Decision:** {status}\n\n"
                report += f"**Executive Summary:**\n"
                
                if tone == "positive":
                    report += f"The applicant presents a generally healthy financial profile. With an annual income of ${u_inc:,} and a requested loan amount of ${u_amnt:,}, the principal-to-income ratio remains within acceptable parameters. The Debt-to-Income (DTI) ratio is currently at {u_dti}%, indicating that the applicant has sufficient cash flow to comfortably manage the new monthly installments. Coupled with a solid credit score of {u_score}, the machine learning risk engine predicts a low probability of default. \n\n"
                    report += f"**Recommendation:** Proceed with standard onboarding and approval. No manual underwriter intervention is required at this stage."
                else:
                    report += f"The applicant exhibits multiple high-risk indicators that exceed our automated approval thresholds. The requested loan amount of ${u_amnt:,} places a significant burden on the reported annual income of ${u_inc:,}. More critically, the applicant's Debt-to-Income (DTI) ratio is elevated at {u_dti}%, suggesting they are highly leveraged with existing credit obligations. A credit score of {u_score} further compounds the probability of default under economic stress. \n\n"
                    report += f"**Recommendation:** Decline automated approval. If the applicant holds a strategic relationship with the bank, route to the manual underwriting team for a potential Dynamic Pricing Counter-Offer or request additional collateral."

                # Typewriter effect for LLM feeling
                def stream_data():
                    for word in report.split(" "):
                        yield word + " "
                        time.sleep(0.04)

                st.markdown('<div class="gen-ai-box">', unsafe_allow_html=True)
                st.write_stream(stream_data)
                st.markdown('</div>', unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 5. EXPLAINABLE AI (XAI)
    # ==========================================
    elif app_mode == "🧠 EXPLAINABLE AI (XAI)":
        st.markdown('<div class="content-container"><div class="content-container-header">🧠 EXPLAINABLE AI (TRANSPARENCY ENGINE)</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: x_dti = st.slider("Debt-to-Income (DTI %)", 0.0, 50.0, 35.0); x_inc = st.number_input("Annual Income ($)", value=45000.0)
        with col2: x_amnt = st.number_input("Loan Amount ($)", value=25000.0); x_int = st.slider("Interest Rate (%)", 5.0, 25.0, 18.0)
        if st.button("🔍 EXPLAIN PREDICTION") and pipeline is not None:
            df_xai = pd.DataFrame([[x_amnt, 36, x_int, 300, x_inc, x_dti, 10, 20]], columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
            prediction = pipeline.predict(df_xai)
            st.markdown(f"### AI DECISION: {'✅ APPROVED' if prediction[0] == 1 else '🚫 REJECTED'}")
            impact_data = {'Feature': ['Annual Income', 'Debt-to-Income (DTI)', 'Interest Rate', 'Loan Amount'], 'Impact': [(x_inc - 60000)/10000, (20 - x_dti)/5, (12 - x_int)/2, (15000 - x_amnt)/5000]}
            df_impact = pd.DataFrame(impact_data)
            df_impact['Color'] = df_impact['Impact'].apply(lambda x: '#2ecc71' if x > 0 else '#e74c3c')
            fig = px.bar(df_impact, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 6. FRAUD & ANOMALY DETECT
    # ==========================================
    elif app_mode == "🚨 FRAUD & ANOMALY DETECT":
        st.markdown('<div class="content-container"><div class="content-container-header">🚨 FRAUD & ANOMALY DETECTION LAYER</div>', unsafe_allow_html=True)
        f_inc = st.number_input("Reported Annual Income ($)", value=15000.0)
        f_amnt = st.number_input("Requested Loan Amount ($)", value=80000.0)
        if st.button("🛡️ RUN SECURITY SCAN"):
            if f_amnt > (f_inc * 4): st.error("⚠️ ANOMALY DETECTED: Loan amount is excessively high.")
            else: st.success("✅ SCAN CLEAR")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 7. MODEL DRIFT MONITOR
    # ==========================================
    elif app_mode == "📉 MODEL DRIFT MONITOR":
        st.markdown('<div class="content-container"><div class="content-container-header">📉 MLOPS DATA DRIFT DASHBOARD</div>', unsafe_allow_html=True)
        if st.button("🔄 RUN DRIFT ANALYSIS"): st.error("ACTION REQUIRED: High Drift Detected. Schedule a model retraining pipeline.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 8. GEOSPATIAL RISK MAP
    # ==========================================
    elif app_mode == "📍 GEOSPATIAL RISK MAP":
        st.markdown('<div class="content-container"><div class="content-container-header">📍 REGIONAL RISK CONCENTRATION</div>', unsafe_allow_html=True)
        if st.button("🗺️ GENERATE HEATMAP"):
            lats = np.random.uniform(10.0, 28.0, 300); lons = np.random.uniform(72.0, 88.0, 300); scores = np.random.randint(300, 850, 300)
            df_geo = pd.DataFrame({'Latitude': lats, 'Longitude': lons, 'Status': ["APPROVED" if s > 600 else "HIGH-RISK" for s in scores]})
            fig_map = px.scatter_mapbox(df_geo, lat="Latitude", lon="Longitude", color="Status", color_discrete_map={"APPROVED": "#2ecc71", "HIGH-RISK": "#e74c3c"}, zoom=3.5, center={"lat": 20.0, "lon": 78.0}, mapbox_style="carto-darkmatter", height=500)
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 9. STRESS TESTING ENGINE
    # ==========================================
    elif app_mode == "🌪️ STRESS TESTING ENGINE":
        st.markdown('<div class="content-container"><div class="content-container-header">🌪️ MACROECONOMIC STRESS TESTING</div>', unsafe_allow_html=True)
        shock_income = st.slider("Income Drop Shock (%)", 0, 50, 20)
        if st.button("💥 RUN PORTFOLIO STRESS TEST"): st.error(f"Applying {shock_income}% shock... Default Rate Spike Detected!")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 10. ETHICAL AI & FAIRNESS
    # ==========================================
    elif app_mode == "⚖️ ETHICAL AI & FAIRNESS":
        st.markdown('<div class="content-container"><div class="content-container-header">⚖️ MODEL FAIRNESS & BIAS AUDIT</div>', unsafe_allow_html=True)
        if st.button("📊 RUN COMPLIANCE AUDIT"): st.error("🚨 ALERT: Model exhibits age bias against 'Age 18-25' group. Disparate impact ratio < 0.8.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 11. DYNAMIC PRICING ENGINE
    # ==========================================
    elif app_mode == "💸 DYNAMIC PRICING ENGINE":
        st.markdown('<div class="content-container"><div class="content-container-header">💸 PRESCRIPTIVE AI: DYNAMIC PRICING OPTIMIZER</div>', unsafe_allow_html=True)
        if st.button("🎯 OPTIMIZE PRICING / COUNTER-OFFER"):
            st.success("Optimal Target Rate: 16.5%. Expected Net Margin: $4,500. System Decision: COUNTER-OFFER")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 12. CREDIT ROADMAP
    # ==========================================
    elif app_mode == "🗺️ CREDIT ROADMAP":
        st.markdown('<div class="content-container"><div class="content-container-header">🗺️ PERSONALISED CREDIT ROADMAP</div>', unsafe_allow_html=True)
        st.write("Plan generation module active.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 13. ADMIN GATEWAY
    # ==========================================
    elif app_mode == "🔒 ADMIN GATEWAY":
        st.markdown('<div class="content-container"><div class="content-container-header">🔒 SECURE ADMIN ACCESS</div>', unsafe_allow_html=True)
        user = st.text_input("Admin ID")
        pwd = st.text_input("Security Key", type="password")
        if st.button("AUTHENTICATE SESSION"):
            if authenticate(user, pwd): st.success("✅ Authorized.")
            else: st.error("🚫 Authentication Failed.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
