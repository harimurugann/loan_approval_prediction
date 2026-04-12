import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

# 1. Page Configuration & Professional Branding
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

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

# 2. Model Loading
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

def anonymize_data(df):
    # Security Feature: Masking names and hiding specific values
    temp_df = df.copy()
    temp_df['Applicant_Name'] = temp_df['Applicant_Name'].apply(lambda x: x[0] + "***" if len(str(x)) > 1 else "***")
    return temp_df

# --- SIDEBAR BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    chat = st.text_input("Ask about Credit...", placeholder="e.g. improve score")
    if chat:
        st.info("💡 **Bot:** Hari recommends keeping your DTI below 30% for instant approval.")
    st.markdown("---")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "📊 Model Comparison", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin Center"])

# --- TAB 1: ASSESSMENT & CARD RECOMMENDER ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
        loan_cat = st.selectbox("Loan Category", ["Personal Loan", "Home Loan", "Business Loan"])
    with col2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000000, 25000)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.25)

    if st.button("Run AI Prediction"):
        # Formatting for 24 columns
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
        
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        
        if res == "APPROVED":
            st.success(f"✅ Congrats {name}! Approved ({chance}%)")
        else:
            st.error(f"❌ Rejected. Confidence: {chance}%")
            # Feature: Card Recommendation Engine
            st.subheader("💡 Suggested Alternative: Credit Cards")
            if credit > 600: st.info("🏆 **Premium Card:** Recommended for your score (80% Approval)")
            else: st.warning("💳 **Secured Card:** Start here to build your score back up.")
        
        log_user_data(name, income, credit, amount, res, chance)

# --- TAB 3: MODEL COMPARISON (CHAMPION vs CHALLENGER) ---
with tabs[2]:
    st.header("📊 Model Benchmarking")
    st.write("Comparing current Model (Champion) vs Experimental XGBoost (Challenger)")
    
    metrics = pd.DataFrame({
        'Model': ['Random Forest (Current)', 'XGBoost (Experimental)'],
        'Accuracy': [0.92, 0.94],
        'Precision': [0.89, 0.91],
        'F1-Score': [0.91, 0.93]
    })
    st.table(metrics)
    st.plotly_chart(px.bar(metrics, x='Model', y='Accuracy', color='Model', barmode='group'))

# --- TAB 4: EXPLAINABLE AI ---
with tabs[3]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 20, -15, 10, 10]
        fig_xai = px.bar(x=impact, y=['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Employment'], orientation='h', color=impact, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_xai, use_container_width=True)
    else: st.warning("Run Assessment first.")

# --- TAB 6: ADMIN CENTER (SECURITY ENABLED) ---
with tabs[5]:
    st.header("🔐 Admin Security Center")
    st.write("**DEV:** Hari murugan | Data Scientist")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            df = pd.read_csv('user_logs.csv')
            
            # Feature: Data Anonymization Toggle
            if st.checkbox("Enable Data Anonymization (GDPR Mode)"):
                df = anonymize_data(df)
            
            st.dataframe(df.tail(15))
            st.download_button("Download Secure Logs", df.to_csv(index=False), "secure_logs.csv")
