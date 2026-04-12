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
    st.toast(f"📧 Alert: Result email queued for {email}", icon="📩")

# --- SIDEBAR: AI CHATBOT & BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    chat_input = st.text_input("Ask Bot...", placeholder="How to fix low credit?")
    if chat_input:
        q = chat_input.lower()
        if "credit" in q: st.info("💡 **Bot:** Pay bills on time and keep card usage below 30%.")
        else: st.write("Ask about 'Credit' or 'DTI'.")
    st.markdown("---")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Processing", "📈 Explainable AI", "🏦 Market Rates", "🔐 Admin Center"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", "Guest User")
        email = st.text_input("Email Address", "user@example.com")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000) # Increased Limit to 1 Crore
        # NEW FEATURE: Loan Category
        loan_cat = st.selectbox("Loan Category", ["Personal Loan", "Home Loan", "Car Loan", "Business Loan", "Education Loan"])
        
    with col2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        # FIX: Increased Loan Amount Limit to 10 Crore
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000) 
        dti = st.slider("Current DTI Ratio", 0.0, 1.0, 0.25)
    
    if st.button("Run AI Prediction"):
        # FIX: ALL 24 COLUMNS to avoid ValueError
        input_df = pd.DataFrame({
            'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
            'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'],
            'debt_to_income_ratio':[dti], 'credit_score':[credit], 'loan_amount':[amount],
            'loan_purpose':[loan_cat], 'interest_rate':[10.5], 'loan_term':[36],
            'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
            'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0],
            'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*dti], 
            'disposable_income':[income/12 - (income/12*dti)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]
        })

        # 2. Prediction logic
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        
        # Guardrail logic
        if credit < 500:
            res = "REJECTED"
            chance = min(chance, 30.0)
            st.error(f"🚫 High Risk: Credit Score {credit} is too low for a {loan_cat}.")
        else:
            res = "APPROVED" if chance >= 50 else "REJECTED"
        
        # Save results for Analytics
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        st.session_state['last_res'] = res
        
        log_user_data(name, income, credit, amount, res, chance)
        simulate_email(name, email, res)

        if res == "APPROVED":
            st.success(f"✅ Final Decision: {res} ({chance}% Confidence) for {loan_cat}")
        else:
            st.error(f"❌ Final Decision: {res} ({chance}% Confidence)")

# --- OTHER TABS (Keeping all previous logic) ---
with tabs[2]:
    st.header("🧠 Explainable AI (XAI)")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 20, -15, 10, 10]
        fig_xai = px.bar(x=impact, y=['Credit Score', 'Income', 'Loan Amount', 'DTI', 'History'], 
                         orientation='h', color=impact, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_xai, use_container_width=True)
    else:
        st.info("Run Assessment first.")

with tabs[4]:
    st.header("🔐 Admin Center")
    st.write("DEV: Hari murugan | Data Scientist")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            st.dataframe(pd.read_csv('user_logs.csv'))
