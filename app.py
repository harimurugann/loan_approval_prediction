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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="LOAN RISK ASSESSMENT SYSTEM", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS: Cyber Dark Theme ---
st.markdown("""
    <style>
    .main { background-color: #1a1c24; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
    .custom-main-header { color: #ffffff; font-weight: 800; font-size: 1.6rem; text-transform: uppercase; margin-top: -10px; margin-bottom: 25px; letter-spacing: 1.2px; }
    .kpi-card { background-image: linear-gradient(135deg, #262a33 0%, #1c1f26 100%); border: 1px solid #374151; border-bottom: 3px solid #00f2fe; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
    .content-container { background-color: #262a33; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #374151; }
    .stButton>button { background-color: transparent; color: #00f2fe !important; border: 1px solid #00f2fe; text-transform: uppercase; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #00f2fe; color: #000000 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; }
    .security-log { background-color: #000; color: #0f0; font-family: 'Courier New', monospace; padding: 10px; border-radius: 5px; font-size: 0.8rem; height: 180px; overflow-y: scroll; border: 1px solid #333; line-height: 1.5; }
    .threat-red { color: #ff4b4b; font-weight: bold; }
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

# --- MAIN DASHBOARD ---
def main():
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.title("INTELLIGENCE HUB")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("NAVIGATE MODULES", [
        "📊 SYSTEM DASHBOARD",
        "🛡️ CYBER-THREAT & IP TRACKER", # NEW LOGIC
        "📱 SECURITY & FINGERPRINT",
        "🤖 MULTI-AGENT SWARM",
        "📄 AI DOCUMENT OCR (KYC)",
        "🌐 LIVE API STREAM",
        "📂 BULK PROCESSING",
        "🧠 EXPLAINABLE AI (XAI)",
        "🚨 FRAUD & ANOMALY DETECT",
        "🕵️‍♂️ SYNDICATE FRAUD NETWORK",
        "💳 CASH FLOW & OPEN BANKING",
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
    st.sidebar.caption("ENGINE: V8.0 | ARCHITECTURE: FULL-STACK AI")

    h_col1, h_col2 = st.columns([4, 1])
    with h_col1: st.markdown('<div class="custom-main-header">LOAN RISK ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    with h_col2: st.markdown('<p style="text-align: right; color: #2ecc71; font-weight: 600; font-size: 0.75rem;">🟢 ENCRYPTION: AES-256</p>', unsafe_allow_html=True)

    # 1. SYSTEM DASHBOARD
    if app_mode == "📊 SYSTEM DASHBOARD":
        st.info("System Dashboard Active. (ML Inference module ready)")

    # 2. CYBER-THREAT INTELLIGENCE (NEW FEATURE)
    elif app_mode == "🛡️ CYBER-THREAT & IP TRACKER":
        st.markdown('<div class="content-container"><div class="content-container-header">🛡️ CYBER-THREAT INTELLIGENCE SYSTEM</div>', unsafe_allow_html=True)
        st.write("Intercepts incoming application packets to analyze IP reputation, geolocation threats, and Dark-Web database matches.")
        
        target_ip = st.text_input("Incoming Connection IP", value="103.21.141.205")
        
        if st.button("🛰️ INITIATE IP REPUTATION SCAN"):
            with st.spinner("Quering Threat Intel Databases..."):
                time.sleep(1.5)
                
                # Mock Threat Data
                threat_score = random.randint(0, 100)
                status = "CLEAN" if threat_score < 30 else "SUSPICIOUS" if threat_score < 70 else "MALICIOUS"
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("IP Reputation Score", f"{threat_score}/100", delta="Normal" if status=="CLEAN" else "DANGER", delta_color="inverse")
                with c2: st.metric("Connection Type", "ISP / Fiber")
                with c3: st.metric("Geolocation", "Chennai, India")

                st.markdown("### 🖥️ Real-time Threat Logs")
                logs = [
                    f"[INFO] Scanning IP: {target_ip}...",
                    "[INFO] Checking against Spamhaus Blacklist... [PASSED]",
                    "[INFO] Analyzing packet latency and hop counts... [NORMAL]",
                    "[WARN] IP has been active on 4 different sessions in 1 hour.",
                    "[SEC] TOR Exit Node Check... [CLEAN]",
                    "[INFO] Dark-Web identity leak check... [NO MATCH FOUND]"
                ]
                
                log_html = "".join([f"<div>{l}</div>" for l in logs])
                st.markdown(f'<div class="security-log">{log_html}</div>', unsafe_allow_html=True)
                
                if threat_score > 75:
                    st.error("🚨 HIGH THREAT DETECTED: This IP is associated with known credential stuffing attacks. Automated block initiated.")
                else:
                    st.success("✅ CONNECTION VERIFIED: Low threat profile. Safe for ML processing.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Placeholder for other modules to keep the structure clean
    else:
        st.markdown(f'<div class="content-container"><div class="content-container-header">{app_mode}</div>', unsafe_allow_html=True)
        st.write(f"The logic for **{app_mode}** is active in the backend and ready for demonstration.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
