import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

# 1. Page Configuration
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;
        padding: 0.6rem 2rem; border: none; transition: 0.3s;
    }
    div.stButton > button:first-child:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(0,204,150,0.3); }
    </style>
    """, unsafe_allow_html=True)

# 2. Load the Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HELPER FUNCTIONS ---
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- SIDEBAR: NEW BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Project Control")
    st.markdown("---")
    # Updated as per your request
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.markdown("---")
    st.subheader("System Status")
    st.success("Model: Operational ✅")
    st.info("Version: 3.1.0 (Ultimate)")
    st.markdown("---")
    st.caption("Loan Eligibility Platform © 2026")

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Individual Assessment", 
    "📂 Bulk Processing", 
    "📈 Analytics & Simulator", 
    "🔐 Admin Control Center"
])

# --- TAB 1: INDIVIDUAL ASSESSMENT (With Logic) ---
with tab1:
    st.header("Single Applicant Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 500000, 55000)
    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 720)
        loan_amount = st.number_input("Loan Amount ($)", 0, 100000, 20000)
    
    # Manual Override Guardrail logic
    if st.button("Analyze Eligibility"):
        prob = model.predict_proba(pd.DataFrame([[30, income, credit_score, loan_amount]], columns=['age', 'annual_income', 'credit_score', 'loan_amount']))[0][1] # Reference logic
        chance = round(prob * 100, 2)
        
        if credit_score < 500:
            res = "REJECTED"
            chance = min(chance, 35.0)
            st.error(f"🚫 High Risk: Credit Score {credit_score} is too low.")
        else:
            res = "APPROVED" if chance >= 50 else "REJECTED"

        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit_score
        log_user_data(name, income, credit_score, loan_amount, res, chance)
        
        if res == "APPROVED": st.success(f"Final Decision: {res} ({chance}% Confidence)")
        else: st.error(f"Final Decision: {res} ({chance}% Confidence)")

# --- TAB 4: ADMIN SECTION (With Password & New Branding) ---
with tab4:
    st.header("🔐 Admin Data Center")
    # Branded Header in Admin
    st.subheader("DEV: Hari murugan | Data Scientist")
    
    pwd = st.text_input("Enter Admin Password", type="password")
    if pwd == "admin123":
        st.success("Authorized Access Granted.")
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv('user_logs.csv')
            
            # Metrics
            c1, c2 = st.columns(2)
            c1.metric("Total Assessments", len(df_logs))
            c2.metric("System Health", "Optimal")
            
            st.dataframe(df_logs, use_container_width=True)
            
            csv_data = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export System Logs", csv_data, "hari_murugan_logs.csv", "text/csv")
        else:
            st.info("No system logs found yet.")
    elif pwd != "":
        st.error("Access Denied.")
