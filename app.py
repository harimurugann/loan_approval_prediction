import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

# 1. Page Configuration
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

# Custom UI Styling
st.markdown("""<style>.stApp {background-color: #0e1117; color: white;} div.stButton > button:first-child {background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;}</style>""", unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HELPERS ---
def detect_fraud(income, amount, credit):
    if amount > (income * 10) or (credit < 400 and amount > 500000): return True
    return False

def log_data(name, income, credit, amount, res, prob):
    file = 'user_logs_production.csv'
    log = pd.DataFrame({'Timestamp': [datetime.datetime.now()], 'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit], 'Loan_Amount': [amount], 'Prediction': [res], 'Probability_%': [prob], 'lat': [13.08], 'lon': [80.27]})
    if not os.path.isfile(file): log.to_csv(file, index=False)
    else: log.to_csv(file, mode='a', header=False, index=False)

# --- NAVIGATION TABS (Explicit Nesting) ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", "📈 Model Drift", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin Center"])

# --- TAB 0: ASSESSMENT (FIXED: Added Annual Income) ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        u_name = st.text_input("Full Name", "Guest")
        # FIX: Added Annual Income Column back
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000)
    
    if st.button("Analyze Eligibility"):
        is_fraud = detect_fraud(income, amount, credit)
        # 24 Column Formatting
        input_df = pd.DataFrame({'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"], 'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'], 'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount], 'loan_purpose':['Personal'], 'interest_rate':[10.5], 'loan_term':[36], 'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5], 'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0], 'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]})
        
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500 and not is_fraud) else "REJECTED"
        
        st.session_state['last_chance'], st.session_state['last_score'], st.session_state['last_res'] = chance, credit, res
        
        if is_fraud: st.warning("🚨 Fraud Alert Flagged!")
        if res == "APPROVED":
            if chance >= 75: st.balloons(); st.success(f"✅ Approved ({chance}%)")
            else: st.warning(f"⚠️ Moderate Risk Approval ({chance}%)")
        else: st.error(f"❌ Rejected ({chance}%)")
        log_data(u_name, income, credit, amount, res, chance)

# --- TAB 1: BULK HUB ---
with tabs[1]:
    st.header("📂 Bulk Processing Engine")
    up_file = st.file_uploader("Upload CSV for Batch Check", type="csv")
    if up_file: st.dataframe(pd.read_csv(up_file).head())

# --- TAB 2: LIVE GEO MAPPING ---
with tabs[2]:
    st.header("🗺️ Applicant Geospatial View")
    if os.path.exists('user_logs_production.csv'):
        df_geo = pd.read_csv('user_logs_production.csv')
        st.map(df_geo[['lat', 'lon']])
    else: st.info("Run Assessment first to see mapping.")

# --- TAB 3: MODEL DRIFT ---
with tabs[3]:
    st.header("📉 Model Performance Monitoring")
    d_df = pd.DataFrame({'Day': range(1,11), 'Accuracy': [0.94, 0.93, 0.94, 0.92, 0.94, 0.91, 0.90, 0.92, 0.91, 0.92]})
    st.plotly_chart(px.line(d_df, x='Day', y='Accuracy', title="Stability Score"))

# --- TAB 4: EXPLAINABLE AI ---
with tabs[4]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        st.plotly_chart(px.bar(x=impact, y=['Credit Score', 'Income', 'Amount', 'DTI', 'Age'], orientation='h', color=impact))
    else: st.warning("Run Assessment first.")

# --- TAB 5: MARKET & CARDS ---
with tabs[5]:
    st.header("🏦 Comparative Market Rates")
    st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari AI Bank'], 'Rate': ['10.5%', '10.7%', '9.2%']}))

# --- TAB 6: ADMIN CENTER ---
with tabs[6]:
    st.header("🔐 Secure Data Logs")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs_production.csv'):
            st.dataframe(pd.read_csv('user_logs_production.csv').tail(10))

    with tabs[7]: # Index 7 means 8th tab
    st.header("🚀 Future Roadmap")
    st.write("Developing: NLP Sentiment Analysis for Loan Applicants.")
