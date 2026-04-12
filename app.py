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

# Custom UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;
        padding: 0.6rem 2rem; border: none; transition: 0.3s;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- LOGGING FUNCTION ---
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- SIDEBAR: CHATBOT & BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    st.write("Ask me how to fix your profile!")
    
    chat_input = st.text_input("Message Bot...", placeholder="e.g. How to improve credit?")
    if chat_input:
        q = chat_input.lower()
        if "credit" in q:
            st.info("💡 **Bot:** Pay bills on time and keep credit card use below 30%.")
        elif "dti" in q:
            st.info("💡 **Bot:** Close small debts or increase monthly income to lower DTI.")
        elif "reject" in q:
            st.info("💡 **Bot:** If rejected, wait 6 months before re-applying and check for errors in your report.")
        else:
            st.write("I'm trained on Credit and DTI. Try asking about those!")

    st.markdown("---")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.success("System: Online ✅")
    st.caption("© 2026 Loan Intel Pro")

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["👤 Assessment", "📂 Bulk Processing", "📈 Analytics & Simulator", "🔐 Admin Center"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tab1:
    st.header("Single Applicant Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Applicant Name", "Guest")
        income = st.number_input("Annual Income ($)", 0, 500000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000, 20000)

    if st.button("Run AI Prediction"):
        # Formatting data for model
        input_data = pd.DataFrame([[30, income, credit, amount]], columns=['age', 'annual_income', 'credit_score', 'loan_amount'])
        prob = model.predict_proba(input_data)[0][1]
        chance = round(prob * 100, 2)
        
        # Guardrail Logic
        if credit < 500:
            res = "REJECTED"
            chance = min(chance, 30.0)
            st.error(f"🚫 Critical Risk: Credit Score {credit} is too low.")
        else:
            res = "APPROVED" if chance >= 50 else "REJECTED"

        # Save to Session State for Analytics Tab
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        st.session_state['last_res'] = res
        
        log_user_data(name, income, credit, amount, res, chance)
        
        if res == "APPROVED":
            st.success(f"✅ Prediction: {res} ({chance}% Confidence)")
        else:
            if credit >= 500: st.error(f"❌ Prediction: {res} ({chance}% Confidence)")

# --- TAB 2: BULK PROCESSING ---
with tab2:
    st.header("📂 Bulk Assessment Engine")
    up_file = st.file_uploader("Upload CSV for Batch Analysis", type="csv")
    if up_file:
        df_bulk = pd.read_csv(up_file)
        st.dataframe(df_bulk.head())
        if st.button("Start Batch Prediction"):
            df_bulk['AI_Result'] = np.where(df_bulk['credit_score'] > 600, "Likely Approved", "Review Required")
            st.success("Batch Prediction Complete!")
            st.dataframe(df_bulk)
    else:
        st.info("Upload a CSV to analyze multiple applicants at once.")

# --- TAB 3: ANALYTICS & SIMULATOR ---
with tab3:
    st.header("📈 Decision Analytics")
    if 'last_chance' in st.session_state:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = st.session_state['last_chance'],
                title = {'text': "Approval Probability %"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00CC96"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("What-If Simulation")
            sim_score = st.slider("Adjust Credit Score for Simulation", 300, 900, int(st.session_state['last_score']))
            st.
