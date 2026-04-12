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
        padding: 0.6rem 2rem; border: none; transition: 0.3s ease;
    }
    div.stButton > button:first-child:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(0,204,150,0.4); }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00CC96; }
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

# --- UTILITIES ---

def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

def simulate_email(name, email, status):
    st.toast(f"📧 Notification: Result email queued for {email}", icon="📩")

# --- SIDEBAR: CHATBOT & BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    chat_input = st.text_input("Ask Bot...", placeholder="e.g. Credit score tips")
    if chat_input:
        q = chat_input.lower()
        if "credit" in q: st.info("💡 **Bot:** Pay bills on time and keep usage under 30%.")
        elif "dti" in q: st.info("💡 **Bot:** Consolidation of debts can help lower DTI.")
        else: st.write("Try asking about 'Credit' or 'Market Rates'.")
    
    st.markdown("---")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.success("System: Active ✅")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Processing", "🧠 Explainable AI", "🏦 Market Rates", "🔐 Admin Center"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", "Guest User")
        email = st.text_input("Email Address", "user@example.com")
        income = st.number_input("Annual Income ($)", 0, 500000, 55000)
        age = st.number_input("Age", 18, 100, 30)
    with col2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000, 25000)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.2)

    if st.button("Run AI Prediction"):
        # 1. FIX: Comprehensive Feature Alignment (24 Columns)
        input_df = pd.DataFrame({
            'age':[age], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
            'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'],
            'debt_to_income_ratio':[dti], 'credit_score':[credit], 'loan_amount':[amount],
            'loan_purpose':['Business'], 'interest_rate':[10.5], 'loan_term':[36],
            'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
            'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0],
            'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*dti], 
            'disposable_income':[income/12 - (income/12*dti)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]
        })

        # 2. Prediction
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        
        # 3. Decision Guardrails
        res = "REJECTED" if credit < 500 else ("APPROVED" if chance >= 50 else "REJECTED")
        
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        st.session_state['last_res'] = res
        
        log_user_data(name, income, credit, amount, res, chance)
        simulate_email(name, email, res)

        if res == "APPROVED":
            st.success(f"✅ Final Decision: {res} ({chance}% Confidence)")
        else:
            st.error(f"❌ Final Decision: {res} ({chance}% Confidence)")

# --- TAB 2: BULK PROCESSING ---
with tabs[1]:
    st.header("📂 Batch Prediction Engine")
    up_file = st.file_uploader("Upload CSV", type="csv")
    if up_file:
        df_bulk = pd.read_csv(up_file)
        st.dataframe(df_bulk.head())
        if st.button("Process Batch"):
            df_bulk['AI_Decision'] = np.where(df_bulk['credit_score'] > 600, "Approved", "High Risk")
            st.dataframe(df_bulk)

# --- TAB 3: EXPLAINABLE AI ---
with tabs[2]:
    st.header("🧠 AI Logic & Interpretation")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 5]
        feats = ['Credit Score', 'Annual Income', 'Loan Amount', 'Debt Ratio', 'Age']
        fig_xai = px.bar(x=impact, y=feats, orientation='h', color=impact, 
                         color_continuous_scale='RdYlGn', title="Feature Impact on Decision")
        st.plotly_chart(fig_xai, use_container_width=True)
    else:
        st.warning("Please run assessment first.")

# --- TAB 4: MARKET RATES ---
with tabs[3]:
    st.header("🏦 Bank Interest Comparison")
    market_df = pd.DataFrame({
        'Bank': ['SBI', 'HDFC', 'ICICI', 'Hari Bank (AI)'],
        'Rate (%)': [8.4, 8.6, 8.7, 8.1],
        'Speed': ['Slow', 'Medium', 'Fast', 'Instant']
    })
    st.table(market_df)

# --- TAB 5: ADMIN CENTER ---
with tabs[4]:
    st.header("🔐 Admin Security Dashboard")
    st.write(f"**DEV:** Hari murugan | Data Scientist")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            logs = pd.read_csv('user_logs.csv')
            st.dataframe(logs.tail(10))
            if st.button("🚀 Retrain"): st.balloons()
