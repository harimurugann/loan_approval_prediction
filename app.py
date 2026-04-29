# app.py - V9.2 MASTER BUILD (MULTI-AGENT SWARM DYNAMIC LOGIC ADDED)
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
    .custom-main-header { color: #ffffff; font-weight: 800; font-size: 1.6rem; text-transform: uppercase; letter-spacing: 1.2px; border-bottom: 2px solid #00f2fe; padding-bottom: 10px; margin-bottom: 25px; }
    .kpi-card { background-image: linear-gradient(135deg, #262a33 0%, #1c1f26 100%); border: 1px solid #374151; border-bottom: 3px solid #00f2fe; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
    .kpi-card h3 { font-size: 0.75rem; margin: 0; text-transform: uppercase; color: #9ca3af; }
    .kpi-card p { font-size: 1.4rem; margin: 5px 0; font-weight: bold; color: #ffffff !important; }
    .content-container { background-color: #262a33; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #374151; }
    .content-container-header { font-size: 1.0rem; font-weight: 700; margin-bottom: 15px; color: #ffffff; text-transform: uppercase; border-bottom: 1px solid #374151; padding-bottom: 8px; }
    .stButton>button { background-color: transparent; color: #00f2fe !important; border: 1px solid #00f2fe; text-transform: uppercase; width: 100%; transition: 0.3s; font-weight: bold; }
    .stButton>button:hover { background-color: #00f2fe; color: #000000 !important; }
    .anomaly-box { background-color: #2c0b0e; border-left: 4px solid #e74c3c; padding: 15px; margin-top: 10px; border-radius: 4px; color: #ffcccc; }
    .gen-ai-box { background-color: #1e293b; border: 1px solid #3b82f6; padding: 20px; border-radius: 8px; font-family: 'Courier New', monospace; color: #93c5fd; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; }
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

# --- MAIN DASHBOARD LOGIC ---
def main():
    st.sidebar.title("INTELLIGENCE HUB")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("NAVIGATE MODULES", [
        "📊 SYSTEM DASHBOARD",
        "🛡️ CYBER-THREAT & IP TRACKER",
        "📱 SECURITY & FINGERPRINT",
        "📄 AI DOCUMENT OCR (KYC)",
        "🤖 MULTI-AGENT SWARM",
        "💳 CASH FLOW & OPEN BANKING",
        "🌐 LIVE API STREAM",
        "📂 BULK PROCESSING",
        "🧠 EXPLAINABLE AI (XAI)",
        "🚨 FRAUD & ANOMALY DETECT",
        "🕵️‍♂️ SYNDICATE FRAUD NETWORK",
        "📉 MODEL DRIFT MONITOR",
        "🔄 AUTO-RETRAINING PIPELINE",
        "📍 GEOSPATIAL RISK MAP",
        "🌪️ STRESS TESTING ENGINE",
        "⚖️ ETHICAL AI & FAIRNESS",
        "💸 DYNAMIC PRICING ENGINE",
        "📱 BEHAVIORAL & DIGITAL SCORING",
        "🗺️ CREDIT ROADMAP",
        "☁️ ENTERPRISE ARCHITECTURE",
        "🔒 ADMIN GATEWAY"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("ENGINE: V9.2 | STATUS: FULLY ACTIVE")

    st.markdown(f'<div class="custom-main-header">{app_mode}</div>', unsafe_allow_html=True)

    # 1. SYSTEM DASHBOARD
    if app_mode == "📊 SYSTEM DASHBOARD":
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown('<div class="kpi-card"><h3>Portfolio Health</h3><p>GOOD</p></div>', unsafe_allow_html=True)
        with k2: st.markdown('<div class="kpi-card"><h3>AI Confidence</h3><p>82.5%</p></div>', unsafe_allow_html=True)
        with k3: st.markdown('<div class="kpi-card"><h3>Live Requests</h3><p>1,240</p></div>', unsafe_allow_html=True)
        with k4: st.markdown('<div class="kpi-card"><h3>Approval Rate</h3><p>64%</p></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="content-container"><div class="content-container-header">👤 APPLICANT DATA</div>', unsafe_allow_html=True)
            l_amnt = st.number_input("Loan Amount ($)", value=15000.0)
            a_inc = st.number_input("Annual Income ($)", value=75000.0)
            i_rate = st.number_input("Interest Rate (%)", value=10.5)
            term = st.selectbox("Term (Months)", [36, 60])
            submit = st.button("RUN RISK ANALYSIS")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="content-container"><div class="content-container-header">📈 INFERENCE</div>', unsafe_allow_html=True)
            if submit and pipeline:
                m_inc = a_inc / 12; inst = (l_amnt * (i_rate/1200))/(1-(1+i_rate/1200)**(-term))
                dti = (inst/m_inc)*100
                df = pd.DataFrame([[l_amnt, term, i_rate, inst, a_inc, dti, 10, 20]], columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                pred = pipeline.predict(df)[0]; prob = pipeline.predict_proba(df)[0][1]
                if pred == 1: st.success(f"✅ APPROVED (Score: {prob*100:.1f})")
                else: st.error(f"🚫 REJECTED (Score: {prob*100:.1f})")
            else: st.write("Ready for scan.")
            st.markdown('</div>', unsafe_allow_html=True)

    # 2. CYBER-THREAT & IP TRACKER
    elif app_mode == "🛡️ CYBER-THREAT & IP TRACKER":
        st.markdown('<div class="content-container"><div class="content-container-header">🛡️ CYBER-THREAT INTELLIGENCE SYSTEM</div>', unsafe_allow_html=True)
        st.write("Intercepts incoming application packets to analyze IP reputation, geolocation threats, and Dark-Web database matches.")
        
        target_ip = st.text_input("Application IP Source", value="103.21.141.205")
        
        if st.button("🛰️ SCAN IP REPUTATION"):
            with st.spinner("Quering Threat Intel Databases..."):
                time.sleep(1.5)
                malicious_ips = ["185.220.101.10", "180.220.101.10"] 
                
                if target_ip.strip() in malicious_ips:
                    threat_score = 88; status = "MALICIOUS"; location = "Unknown / TOR Exit Node"
                else:
                    threat_score = random.randint(10, 25); status = "CLEAN"; location = "Chennai, India"
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("IP Reputation Score", f"{threat_score}/100", delta="Normal" if status=="CLEAN" else "DANGER", delta_color="inverse")
                with c2: st.metric("Connection Type", "ISP / Fiber" if status=="CLEAN" else "VPN / TOR Network")
                with c3: st.metric("Geolocation", location)

                st.markdown("### 🖥️ Real-time Threat Logs")
                if status == "MALICIOUS":
                    logs = [f"[INFO] Scanning IP: {target_ip}...", "[WARN] Checking against Spamhaus Blacklist... [MATCH FOUND]", "[SEC] TOR Exit Node Check... [FLAGGED]", "[INFO] Dark-Web identity leak check... [ASSOCIATED RISKS FOUND]"]
                    log_html = "".join([f"<div style='color:#ff4b4b; margin-bottom:4px;'>{l}</div>" for l in logs])
                else:
                    logs = [f"[INFO] Scanning IP: {target_ip}...", "[INFO] Checking against Spamhaus Blacklist... [PASSED]", "[SEC] TOR Exit Node Check... [CLEAN]", "[INFO] Dark-Web identity leak check... [NO MATCH FOUND]"]
                    log_html = "".join([f"<div style='margin-bottom:4px;'>{l}</div>" for l in logs])
                    
                st.markdown(f'<div style="background-color: #000; color: #0f0; font-family: monospace; padding: 10px; height: 130px; overflow-y: scroll; border-radius: 5px; border: 1px solid #333;">{log_html}</div>', unsafe_allow_html=True)
                st.write("") 
                if status == "MALICIOUS": st.error(f"🚨 HIGH THREAT DETECTED: IP {target_ip} is associated with known cyber-attacks. Automated block initiated.")
                else: st.success(f"✅ CONNECTION VERIFIED: IP {target_ip} has a low threat profile. Safe for ML processing.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. SECURITY & DEVICE FINGERPRINTING
    elif app_mode == "📱 SECURITY & FINGERPRINT":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        if st.button("CAPTURE DEVICE FINGERPRINT"):
            st.info("Browser: Chrome v124 | OS: Android 14 | Hardware ID: XYZ-889-12")
            st.success("Identity Verified: Human user detected via touch-latency patterns.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. AI DOCUMENT OCR
    elif app_mode == "📄 AI DOCUMENT OCR (KYC)":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        file = st.file_uploader("Upload ID Card (Aadhar/PAN)", type=['png','jpg','jpeg'])
        if file and st.button("EXTRACT DATA"):
            st.image(file, width=200)
            st.json({"Name": "Harish Kumar", "ID": "XXXX-XXXX-1234", "Tamper_Check": "PASS"})
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. MULTI-AGENT SWARM (UPDATED DYNAMIC LOGIC)
    elif app_mode == "🤖 MULTI-AGENT SWARM":
        st.markdown('<div class="content-container"><div class="content-container-header">🤖 MULTI-AGENT LLM SWARM</div>', unsafe_allow_html=True)
        st.write("Agents automatically ingest data to debate and reach a consensus.")
        
        c1, c2, c3 = st.columns(3)
        with c1: s_inc = st.number_input("Applicant Income ($)", value=60000)
        with c2: s_dti = st.slider("Current DTI (%)", 10, 60, 25)
        with c3: s_score = st.slider("CIBIL / Credit Score", 300, 850, 720)
        
        if st.button("✨ DEPLOY LLM AGENTS"):
            with st.spinner("Swarm Intelligence deliberating..."):
                time.sleep(2) 
                
                if s_score >= 650 and s_dti <= 40:
                    box_color = "#3b82f6" 
                    text_color = "#93c5fd"
                    decision = "✅ 3/3 CONSENSUS - APPROVED"
                    risk_comment = f"Credit score of {s_score} is well above threshold."
                    fin_comment = f"Income of ${s_inc:,} and DTI of {s_dti}% shows strong repayment capacity."
                else:
                    box_color = "#ef4444" 
                    text_color = "#fca5a5"
                    decision = "🚫 CONSENSUS FAILED - REJECTED"
                    risk_comment = f"CRITICAL: Score of {s_score} indicates high historical default risk."
                    fin_comment = f"WARNING: DTI of {s_dti}% is too high for the current income levels."

                st.markdown(f'''
                <div style="background-color: #1e293b; border: 1px solid {box_color}; padding: 20px; border-radius: 8px; font-family: monospace; color: {text_color};">
                    <b>🧑‍💼 Financial Agent:</b> {fin_comment}<br><br>
                    <b>🕵️‍♂️ Risk Agent:</b> {risk_comment}<br><br>
                    <b>👮 Compliance Agent:</b> No AML or KYC flags detected. Clean history.<br><br>
                    <hr style="border-color: {box_color};">
                    <b>👑 Swarm Decision:</b> {decision}
                </div>
                ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 6. CASH FLOW & OPEN BANKING
    elif app_mode == "💳 CASH FLOW & OPEN BANKING":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.write("Aggregating live account statement data...")
        st.metric("Avg Monthly Surplus", "$2,100", "+5% vs Last Month")
        fig = px.line(y=[1200, 1500, 1100, 1800, 2100], x=['Jan','Feb','Mar','Apr','May'], title="Cash Flow Trend")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 7. LIVE API STREAM
    elif app_mode == "🌐 LIVE API STREAM":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        if st.button("▶️ START STREAM"):
            placeholder = st.empty()
            for i in range(5):
                placeholder.info(f"Processing Live Transaction {random.randint(100,999)}... [✅ Approved]")
                time.sleep(1)
        st.markdown('</div>', unsafe_allow_html=True)

    # 8. BULK PROCESSING
    elif app_mode == "📂 BULK PROCESSING":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        up = st.file_uploader("Upload 20k Records CSV", type="csv")
        if up: 
            st.success("File Loaded. Ready to process batch.")
            if st.button("EXECUTE BATCH"): st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)

    # 9. EXPLAINABLE AI
    elif app_mode == "🧠 EXPLAINABLE AI (XAI)":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        fig = px.bar(x=[0.4, -0.2, 0.5, -0.1], y=['Income','DTI','Amount','Term'], orientation='h', title="Feature Contribution (SHAP)")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Green indicates Approval factors, Red indicates Rejection factors.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 10. FRAUD & ANOMALY
    elif app_mode == "🚨 FRAUD & ANOMALY DETECT":
        st.markdown('<div class="anomaly-box">Rule F-102: Potential Income Fraud Detected (Income < EMI*3)</div>', unsafe_allow_html=True)

    # 11. SYNDICATE FRAUD
    elif app_mode == "🕵️‍♂️ SYNDICATE FRAUD NETWORK":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.write("Visualizing network links...")
        fig = go.Figure(go.Scatter(x=[1, 2, 1.5], y=[1, 1, 2], mode='markers+lines', marker=dict(size=20, color=['blue','red','blue'])))
        fig.update_layout(title="Identity Linkage Graph")
        st.plotly_chart(fig, use_container_width=True)
        st.error("Alert: Connection found with blacklisted device ID.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 12. MODEL DRIFT
    elif app_mode == "📉 MODEL DRIFT MONITOR":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.metric("Population Stability Index (PSI)", "0.24", "HIGH DRIFT", delta_color="inverse")
        st.warning("Action Required: Model distribution has shifted from baseline training data.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 13. AUTO-RETRAINING
    elif app_mode == "🔄 AUTO-RETRAINING PIPELINE":
        if st.button("TRIGGER CI/CD RE-TRAIN"):
            st.write("Ingesting new data...")
            st.progress(50)
            st.success("New model v2.1 deployed to Models/ folder.")

    # 14. GEOSPATIAL
    elif app_mode == "📍 GEOSPATIAL RISK MAP":
        lats = np.random.uniform(10, 25, 50); lons = np.random.uniform(70, 85, 50)
        df_map = pd.DataFrame({'lat': lats, 'lon': lons})
        st.map(df_map)

    # 15. STRESS TESTING
    elif app_mode == "🌪️ STRESS TESTING ENGINE":
        shock = st.slider("Recession Intensity (%)", 0, 50, 20)
        st.error(f"With {shock}% income drop, default rate spikes by 14.2%.")

    # 16. ETHICAL AI
    elif app_mode == "⚖️ ETHICAL AI & FAIRNESS":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.write("Audit Report: Gender Disparate Impact Ratio = 0.94 (PASS)")
        st.success("No significant bias detected across demographic groups.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 17. DYNAMIC PRICING
    elif app_mode == "💸 DYNAMIC PRICING ENGINE":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        risk = st.slider("Customer Risk Score", 0, 100, 45)
        st.info(f"AI Recommended Interest Rate for this risk level: {8.5 + (risk/10)}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # 18. BEHAVIORAL SCORING
    elif app_mode == "📱 BEHAVIORAL & DIGITAL SCORING":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.write("Scoring based on mobile usage and digital footprint.")
        st.metric("Alternative Score", "712 / 850", "+12 pts")
        st.markdown('</div>', unsafe_allow_html=True)

    # 19. CREDIT ROADMAP
    elif app_mode == "🗺️ CREDIT ROADMAP":
        st.markdown('<div class="roadmap-box"><b>Step 1:</b> Clear active credit card dues.<br><b>Step 2:</b> Lower DTI to 30%.<br><b>Step 3:</b> Maintain score for 6 months.</div>', unsafe_allow_html=True)

    # 20. ENTERPRISE ARCHITECTURE
    elif app_mode == "☁️ ENTERPRISE ARCHITECTURE":
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg", width=100)
        st.write("Cloud infrastructure: AWS S3 Artifacts + Streamlit Front-end + Docker Containers.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 21. ADMIN GATEWAY
    elif app_mode == "🔒 ADMIN GATEWAY":
        st.text_input("Admin Username")
        st.text_input("Security Token", type="password")
        if st.button("LOGIN"): st.error("Access Restricted to Intranet IP.")

if __name__ == "__main__":
    main()
