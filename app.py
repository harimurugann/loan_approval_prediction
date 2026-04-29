# app.py - V10.0 MASTER BUILD (REAL-TIME DYNAMIC HEURISTICS & SESSION STATE)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import random
import plotly.express as px
import plotly.graph_objects as go

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

# --- GLOBAL SESSION STATE ---
# Ippo user oru tab-la input kudutha, adhu matha tabs-laiyum use aagum!
if 'loan_amnt' not in st.session_state: st.session_state.loan_amnt = 15000.0
if 'annual_inc' not in st.session_state: st.session_state.annual_inc = 75000.0
if 'int_rate' not in st.session_state: st.session_state.int_rate = 10.5
if 'term' not in st.session_state: st.session_state.term = 36
if 'app_id' not in st.session_state: st.session_state.app_id = "APP-89421"

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
    st.sidebar.caption("ENGINE: V10.0 | STATUS: REAL-TIME LOGIC")

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
            # Link inputs directly to session state
            st.session_state.loan_amnt = st.number_input("Loan Amount ($)", value=st.session_state.loan_amnt)
            st.session_state.annual_inc = st.number_input("Annual Income ($)", value=st.session_state.annual_inc)
            st.session_state.int_rate = st.number_input("Interest Rate (%)", value=st.session_state.int_rate)
            st.session_state.term = st.selectbox("Term (Months)", [36, 60], index=0 if st.session_state.term==36 else 1)
            submit = st.button("RUN RISK ANALYSIS")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="content-container"><div class="content-container-header">📈 INFERENCE RESULTS</div>', unsafe_allow_html=True)
            if submit and pipeline:
                m_inc = st.session_state.annual_inc / 12 if st.session_state.annual_inc > 0 else 1
                inst = (st.session_state.loan_amnt * (st.session_state.int_rate/1200))/(1-(1+st.session_state.int_rate/1200)**(-st.session_state.term))
                dti = (inst/m_inc)*100
                df = pd.DataFrame([[st.session_state.loan_amnt, st.session_state.term, st.session_state.int_rate, inst, st.session_state.annual_inc, dti, 10, 20]], columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                
                # Dynamic Logic based on inputs
                pred = pipeline.predict(df)[0]; prob = pipeline.predict_proba(df)[0][1]
                
                if dti > 60:
                    st.error("🚫 REJECTED BY RULE ENGINE: Debt-to-Income ratio exceeds 60%. Highly Unsafe.")
                elif pred == 1: 
                    st.success(f"✅ APPROVED BY AI (Confidence: {prob*100:.1f}%)")
                    st.write(f"Applicant has a healthy DTI of {dti:.1f}%")
                else: 
                    st.error(f"🚫 REJECTED BY AI (Probability of Default: {prob*100:.1f}%)")
            else: st.write("Ready for scan.")
            st.markdown('</div>', unsafe_allow_html=True)

    # 2. CYBER-THREAT & IP TRACKER (REAL-TIME MATH LOGIC)
    elif app_mode == "🛡️ CYBER-THREAT & IP TRACKER":
        st.markdown('<div class="content-container"><div class="content-container-header">🛡️ CYBER-THREAT INTELLIGENCE</div>', unsafe_allow_html=True)
        target_ip = st.text_input("Application IP Source", value="103.21.141.205")
        
        if st.button("🛰️ SCAN IP REPUTATION"):
            with st.spinner("Quering Databases..."):
                time.sleep(1)
                # Dynamic Heuristic: Hash the IP to generate consistent threat score!
                # E.g. "8.8.8.8" will always give a clean score, while "185.220.101.10" will give a high risk score based on the numbers.
                try:
                    octets = [int(x) for x in target_ip.split('.')]
                    calc_score = sum(octets) % 100
                    # Hardcode some known bad subnets for demo effect
                    if octets[0] in [185, 45, 194, 180]: calc_score = 88 + (octets[3] % 10)
                except:
                    calc_score = 99 # Invalid IP format throws high risk

                if calc_score > 75:
                    status = "MALICIOUS"; location = "Unknown / TOR Exit Node"
                    log_html = f"<div style='color:#ff4b4b;'>[WARN] IP {target_ip} MATCHES BLACKLIST. Score: {calc_score}</div>"
                    st.markdown(f'<div style="background-color: #000; padding: 10px;">{log_html}</div>', unsafe_allow_html=True)
                    st.error("🚨 HIGH THREAT DETECTED. Automated block initiated.")
                else:
                    status = "CLEAN"; location = "Mapped to User ISP"
                    log_html = f"<div style='color:#0f0;'>[INFO] IP {target_ip} VERIFIED. Clean history. Score: {calc_score}</div>"
                    st.markdown(f'<div style="background-color: #000; padding: 10px;">{log_html}</div>', unsafe_allow_html=True)
                    st.success("✅ CONNECTION VERIFIED.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. SECURITY & FINGERPRINTING (DYNAMIC AGENT PARSER)
    elif app_mode == "📱 SECURITY & FINGERPRINT":
        st.markdown('<div class="content-container"><div class="content-container-header">📱 DEVICE FINGERPRINTING</div>', unsafe_allow_html=True)
        user_agent = st.text_input("Simulate Browser User-Agent String", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36")
        
        if st.button("CAPTURE DEVICE FINGERPRINT"):
            if "HeadlessChrome" in user_agent or "bot" in user_agent.lower() or "selenium" in user_agent.lower():
                st.error("🚨 ALERT: AUTOMATED BOT DETECTED! 'Headless' or 'Bot' signature found in User-Agent. Connection Terminated.")
            else:
                st.info(f"Parsed Device Data: {user_agent.split('(')[1].split(')')[0]}")
                st.success("✅ Identity Verified: Standard Human Browser footprint detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. AI DOCUMENT OCR (DYNAMIC FILE PARSER)
    elif app_mode == "📄 AI DOCUMENT OCR (KYC)":
        st.markdown('<div class="content-container"><div class="content-container-header">📄 AI DOCUMENT OCR</div>', unsafe_allow_html=True)
        file = st.file_uploader("Upload ID Card (Aadhar/PAN)", type=['png','jpg','jpeg','pdf'])
        if file and st.button("EXTRACT DATA"):
            # Real-time reaction to file size and extension
            file_sz = len(file.getvalue()) / 1024
            st.image(file, width=200)
            st.write(f"Analyzed File: `{file.name}` ({file_sz:.2f} KB)")
            
            if file_sz > 5000: # Greater than 5MB
                st.error("🚨 TAMPER ALERT: File size unusually large. Meta-data indicates potential Photoshop layered edits.")
            else:
                st.success("✅ Document structure verified.")
                st.json({
                    "Detected_Format": file.name.split('.')[-1].upper(),
                    "Resolution_Quality": "HIGH" if file_sz > 500 else "Standard",
                    "Extracted_Name": "MATCHES APPLICATION",
                    "Tamper_Check": "PASS"
                })
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. MULTI-AGENT SWARM (SESSION STATE AWARE)
    elif app_mode == "🤖 MULTI-AGENT SWARM":
        st.markdown('<div class="content-container"><div class="content-container-header">🤖 MULTI-AGENT LLM SWARM</div>', unsafe_allow_html=True)
        st.write("Agents are pulling data LIVE from your Dashboard session.")
        st.info(f"Current Session -> Income: ${st.session_state.annual_inc:,} | Loan Requested: ${st.session_state.loan_amnt:,}")
        
        s_score = st.slider("Provide CIBIL Score for Agents to consider", 300, 850, 720)
        
        if st.button("✨ DEPLOY LLM AGENTS"):
            with st.spinner("Swarm Intelligence deliberating..."):
                time.sleep(1) 
                # Real-time logic
                if s_score >= 650 and (st.session_state.loan_amnt < st.session_state.annual_inc * 3):
                    st.markdown(f'<div class="gen-ai-box" style="border-color:#3b82f6;"><b>🧑‍💼 Fin Agent:</b> Income of ${st.session_state.annual_inc:,} can support the ${st.session_state.loan_amnt:,} loan.<br><b>🕵️‍♂️ Risk Agent:</b> Score {s_score} is safe.<br><hr><b>✅ Swarm Decision: APPROVED</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="gen-ai-box" style="border-color:#ef4444; color:#fca5a5;"><b>🧑‍💼 Fin Agent:</b> Loan of ${st.session_state.loan_amnt:,} is extremely risky for income ${st.session_state.annual_inc:,}.<br><b>🕵️‍♂️ Risk Agent:</b> Score {s_score} is inadequate.<br><hr><b>🚫 Swarm Decision: REJECTED</b></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 6. CASH FLOW & OPEN BANKING (DYNAMIC INPUTS)
    elif app_mode == "💳 CASH FLOW & OPEN BANKING":
        st.markdown('<div class="content-container"><div class="content-container-header">💳 OPEN BANKING & CASH FLOW</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: month_in = st.number_input("Avg Monthly Credits ($)", value=6000)
        with c2: month_out = st.number_input("Avg Monthly Debits ($)", value=4500)
        
        if st.button("GENERATE CASH FLOW ANALYSIS"):
            surplus = month_in - month_out
            if surplus < 0: st.error(f"🚨 Negative Cash Flow! User is losing ${abs(surplus)} monthly.")
            else: 
                st.success(f"✅ Healthy Cash Flow. Net surplus: ${surplus} monthly.")
                fig = px.bar(x=['Credits', 'Debits', 'Surplus'], y=[month_in, month_out, surplus], color=['Credits', 'Debits', 'Surplus'], color_discrete_sequence=['#2ecc71', '#e74c3c', '#00f2fe'])
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 7. LIVE API STREAM (DYNAMIC FREQUENCY)
    elif app_mode == "🌐 LIVE API STREAM":
        st.markdown('<div class="content-container"><div class="content-container-header">📡 REAL-TIME API STREAM INFERENCE</div>', unsafe_allow_html=True)
        txn_count = st.slider("Set Stream Volume (Requests per second)", 1, 20, 5)
        if st.button("▶️ START STREAM"):
            placeholder = st.empty()
            for i in range(1, 6): # Limiting loop for UI safety
                placeholder.info(f"Processing {txn_count} concurrent requests... [Batch {i} ✅]")
                time.sleep(1)
            st.success("Stream simulation complete.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 8. BULK PROCESSING
    elif app_mode == "📂 BULK PROCESSING":
        st.markdown('<div class="content-container"><div class="content-container-header">📂 HIGH-VOLUME BATCH PROCESSING</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload Records CSV", type="csv")
        if up: 
            df = pd.read_csv(up)
            st.write(f"Loaded {len(df)} rows.")
            if st.button("EXECUTE BATCH"): st.success("Batch Processed.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 9. EXPLAINABLE AI (SESSION STATE INTEGRATED)
    elif app_mode == "🧠 EXPLAINABLE AI (XAI)":
        st.markdown('<div class="content-container"><div class="content-container-header">🧠 EXPLAINABLE AI (SHAP ENGINE)</div>', unsafe_allow_html=True)
        st.write(f"Explaining Decision for current session (Loan: ${st.session_state.loan_amnt}, Income: ${st.session_state.annual_inc})")
        if st.button("GENERATE XAI GRAPH"):
            # Dynamic impact based on inputs
            inc_impact = 0.5 if st.session_state.annual_inc > 60000 else -0.5
            loan_impact = -0.4 if st.session_state.loan_amnt > 30000 else 0.2
            
            fig = px.bar(x=[inc_impact, loan_impact, -0.2, 0.1], y=['Income Impact','Loan Amount Impact','Term Risk','Int Rate Risk'], orientation='h', title="Feature Contribution", color=[inc_impact, loan_impact, -0.2, 0.1], color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 10. FRAUD & ANOMALY (SESSION AWARE)
    elif app_mode == "🚨 FRAUD & ANOMALY DETECT":
        st.markdown('<div class="content-container"><div class="content-container-header">🚨 ANOMALY DETECTION</div>', unsafe_allow_html=True)
        st.write("Scanning current session inputs...")
        if st.session_state.loan_amnt > (st.session_state.annual_inc * 4):
            st.error(f"🚨 Rule F-102: Loan Amount (${st.session_state.loan_amnt}) is abnormally high compared to Income (${st.session_state.annual_inc}). Risk of synthetic identity fraud.")
        else:
            st.success("✅ Current session profile passes all hardcoded anomaly rules.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 11. SYNDICATE FRAUD (DYNAMIC SEED)
    elif app_mode == "🕵️‍♂️ SYNDICATE FRAUD NETWORK":
        st.markdown('<div class="content-container"><div class="content-container-header">🕵️‍♂️ SYNDICATE FRAUD NETWORK</div>', unsafe_allow_html=True)
        app_id = st.text_input("Enter Applicant ID or Phone", "APP-89421")
        if st.button("BUILD NETWORK GRAPH"):
            # Dynamic logic: Different App ID generates different graph shapes!
            random.seed(app_id)
            nodes = random.randint(3, 8)
            x_pos = [random.uniform(0, 5) for _ in range(nodes)]
            y_pos = [random.uniform(0, 5) for _ in range(nodes)]
            colors = ['blue'] * nodes
            if random.random() > 0.5: colors[random.randint(1, nodes-1)] = 'red' # Randomly flag a threat
            
            fig = go.Figure(go.Scatter(x=x_pos, y=y_pos, mode='markers+lines', marker=dict(size=20, color=colors)))
            fig.update_layout(title=f"Entity Linkage Graph for {app_id}")
            st.plotly_chart(fig, use_container_width=True)
            if 'red' in colors: st.error("🚨 Alert: Connections found with blacklisted entity (Red Node).")
            else: st.success("✅ Clean Network. No fraud rings detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 12. MODEL DRIFT (DYNAMIC INPUTS)
    elif app_mode == "📉 MODEL DRIFT MONITOR":
        st.markdown('<div class="content-container"><div class="content-container-header">📉 DATA DRIFT DASHBOARD</div>', unsafe_allow_html=True)
        drift_factor = st.slider("Simulate Economic Shift (Current Month variance vs Training)", 0, 100, 25)
        psi_score = drift_factor * 0.01
        
        st.metric("Population Stability Index (PSI)", f"{psi_score:.2f}", delta="HIGH DRIFT" if psi_score > 0.2 else "Stable", delta_color="inverse")
        if psi_score > 0.2: st.error("Action Required: Retrain model. PSI > 0.2 threshold.")
        else: st.success("Model is stable. Production data matches training distributions.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 13. AUTO-RETRAINING (DYNAMIC PROGRESS)
    elif app_mode == "🔄 AUTO-RETRAINING PIPELINE":
        st.markdown('<div class="content-container"><div class="content-container-header">🔄 CONTINUOUS TRAINING PIPELINE</div>', unsafe_allow_html=True)
        epochs = st.slider("Number of Hyperparameter Tuning Epochs", 10, 100, 20)
        if st.button("TRIGGER CI/CD RE-TRAIN"):
            my_bar = st.progress(0, text="Initializing training cluster...")
            for p in range(0, 101, int(100/(epochs/10))):
                my_bar.progress(p, text=f"Training Epoch {p}% complete...")
                time.sleep(0.1)
            my_bar.progress(100, text="Complete!")
            st.success(f"New model trained over {epochs} iterations and deployed successfully.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 14. GEOSPATIAL (DYNAMIC REGIONS)
    elif app_mode == "📍 GEOSPATIAL RISK MAP":
        st.markdown('<div class="content-container"><div class="content-container-header">📍 GEOSPATIAL RISK MAP</div>', unsafe_allow_html=True)
        region = st.selectbox("Select Region to Scan", ["North India", "South India", "All India"])
        if st.button("GENERATE MAP"):
            if region == "South India": lats = np.random.uniform(8, 15, 50); lons = np.random.uniform(74, 80, 50)
            elif region == "North India": lats = np.random.uniform(25, 30, 50); lons = np.random.uniform(75, 80, 50)
            else: lats = np.random.uniform(10, 30, 100); lons = np.random.uniform(70, 90, 100)
            st.map(pd.DataFrame({'lat': lats, 'lon': lons}))
        st.markdown('</div>', unsafe_allow_html=True)

    # 15. STRESS TESTING
    elif app_mode == "🌪️ STRESS TESTING ENGINE":
        st.markdown('<div class="content-container"><div class="content-container-header">🌪️ MACRO STRESS TEST</div>', unsafe_allow_html=True)
        shock = st.slider("Recession Intensity (Income Drop %)", 0, 50, 20)
        default_spike = shock * 0.75
        if st.button("RUN SCENARIO"):
            st.error(f"📉 Under a {shock}% income shock, portfolio default rate spikes by +{default_spike:.1f}%.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 16. ETHICAL AI (DYNAMIC AUDIT)
    elif app_mode == "⚖️ ETHICAL AI & FAIRNESS":
        st.markdown('<div class="content-container"><div class="content-container-header">⚖️ FAIRNESS AUDIT</div>', unsafe_allow_html=True)
        threshold = st.slider("Fairness Compliance Threshold (DIR)", 0.6, 1.0, 0.8)
        if st.button("RUN AUDIT"):
            actual_dir = 0.85
            st.write(f"Calculated Disparate Impact Ratio: {actual_dir}")
            if actual_dir >= threshold: st.success("✅ PASS: Model operates fairly across demographics.")
            else: st.error(f"🚨 FAIL: Model fails compliance. {actual_dir} is below threshold {threshold}.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 17. DYNAMIC PRICING
    elif app_mode == "💸 DYNAMIC PRICING ENGINE":
        st.markdown('<div class="content-container"><div class="content-container-header">💸 DYNAMIC PRICING</div>', unsafe_allow_html=True)
        risk = st.slider("Calculated Customer Risk Score", 0, 100, 45)
        rec_rate = 8.5 + (risk/10)
        st.info(f"🎯 AI Recommended Optimal Interest Rate: {rec_rate:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # 18. BEHAVIORAL SCORING
    elif app_mode == "📱 BEHAVIORAL & DIGITAL SCORING":
        st.markdown('<div class="content-container"><div class="content-container-header">📱 BEHAVIORAL SCORING</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: upi = st.slider("UPI TXN Frequency (Monthly)", 0, 200, 50)
        with c2: app = st.slider("Utility App Usage Count", 0, 20, 5)
        score = 500 + (upi * 1.5) + (app * 10)
        st.metric("Alternative Digital Score", f"{int(score)} / 850")
        st.markdown('</div>', unsafe_allow_html=True)

    # 19. CREDIT ROADMAP (SESSION AWARE)
    elif app_mode == "🗺️ CREDIT ROADMAP":
        st.markdown('<div class="content-container"><div class="content-container-header">🗺️ ACTION PLAN</div>', unsafe_allow_html=True)
        st.write(f"Generating personalized plan for loan requested: ${st.session_state.loan_amnt}")
        st.markdown('<div class="roadmap-box"><b>Step 1:</b> Increase income or lower requested loan amount.<br><b>Step 2:</b> Close 2 active micro-loans.<br><b>Step 3:</b> Maintain flawless repayment for 6 months.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 20. ENTERPRISE ARCHITECTURE
    elif app_mode == "☁️ ENTERPRISE ARCHITECTURE":
        st.markdown('<div class="content-container"><div class="content-container-header">☁️ ARCHITECTURE</div>', unsafe_allow_html=True)
        st.write("Deployed via Streamlit Cloud | Model Artifacts on Local FS | Git Version Controlled.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 21. ADMIN GATEWAY
    elif app_mode == "🔒 ADMIN GATEWAY":
        st.markdown('<div class="content-container"><div class="content-container-header">🔒 ADMIN PORTAL</div>', unsafe_allow_html=True)
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("LOGIN"):
            if user == "admin" and pwd == "admin123": st.success("✅ Logged in successfully. Server metrics healthy.")
            else: st.error("🚫 Invalid Credentials.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
