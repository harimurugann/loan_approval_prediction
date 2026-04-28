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
    .security-log { background-color: #000; color: #0f0; font-family: 'Courier New', Courier, monospace; padding: 10px; border-radius: 5px; font-size: 0.8rem; height: 150px; overflow-y: scroll; border: 1px solid #333; }
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

def main():
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.title("INTELLIGENCE HUB")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("NAVIGATE MODULES", [
        "📊 SYSTEM DASHBOARD",
        "🛡️ SECURITY & DEVICE FINGERPRINT", # NEW
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
    st.sidebar.caption("ENGINE: V7.0 | ARCHITECTURE: CYBER-AI")

    h_col1, h_col2 = st.columns([4, 1])
    with h_col1: st.markdown('<div class="custom-main-header">LOAN RISK ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    with h_col2: st.markdown('<p style="text-align: right; color: #2ecc71; font-weight: 600; font-size: 0.75rem;">🟢 STATUS: ONLINE</p>', unsafe_allow_html=True)

    # 1. DASHBOARD (Minimal Logic for Flow)
    if app_mode == "📊 SYSTEM DASHBOARD":
        st.write("Dashboard Active.")

    # 2. SECURITY & DEVICE FINGERPRINTING (NEW)
    elif app_mode == "🛡️ SECURITY & DEVICE FINGERPRINT":
        st.markdown('<div class="content-container"><div class="content-container-header">🛡️ CYBER-SECURITY: DEVICE FINGERPRINTING</div>', unsafe_allow_html=True)
        st.write("This module analyzes hardware signatures and connection metadata to ensure the application is coming from a trusted human user, not a bot or a masked emulator.")
        
        if st.button("🛰️ SCAN SESSION METADATA"):
            with st.spinner("Intercepting Session Headers..."):
                time.sleep(1)
                st.markdown("### 🖥️ Device Signature Output")
                
                # Mock Device Data Logic
                device_data = {
                    "Platform": "Mobile/Android-v14",
                    "Screen Resolution": "1080x2400",
                    "Battery Status": "92% (Discharging)",
                    "IP Location": "Vazhapadi, Tamil Nadu, India",
                    "VPN/Proxy Detected": "False",
                    "Touch Input Consistency": "98% (Human Pattern)"
                }
                
                c1, c2 = st.columns(2)
                for i, (k, v) in enumerate(device_data.items()):
                    with (c1 if i % 2 == 0 else c2):
                        st.info(f"**{k}**: {v}")

                st.markdown("### 🕵️ Security Audit Log")
                log_entries = [
                    "[INFO] Session initialized from trusted IP range.",
                    "[SEC] No Debugger tools detected in browser.",
                    "[SEC] Fonts and Canvas fingerprinting matches user history.",
                    "[WARN] High-frequency request pattern check: PASSED.",
                    "[INFO] Hardware Integrity: VERIFIED."
                ]
                
                log_html = "".join([f"<div>{entry}</div>" for entry in log_entries])
                st.markdown(f'<div class="security-log">{log_html}</div>', unsafe_allow_html=True)
                
                st.success("🔒 DEVICE TRUST SCORE: 99/100 (Safe Connection)")
        st.markdown('</div>', unsafe_allow_html=True)

    # (Other features 3-19 remain mapped to their logic placeholders)
    elif app_mode == "📄 AI DOCUMENT OCR (KYC)":
        st.info("Document OCR Engine Ready.")
    elif app_mode == "🤖 MULTI-AGENT SWARM":
        st.info("AI Agents Online.")
    elif app_mode == "🔄 AUTO-RETRAINING PIPELINE":
        st.info("MLOps Pipeline Ready.")
    else:
        st.info(f"{app_mode} module selected.")

if __name__ == "__main__":
    main()
