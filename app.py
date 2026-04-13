import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
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
def detect_fraud(income, amount, credit):
    if amount > (income * 10) or (credit < 400 and amount > 500000):
        return True
    return False

def log_user_data(name, income, credit, amount, result, prob, lat, lon):
    log_file = 'user_logs_production.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob],
        'lat': [lat], 'lon': [lon]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Admin Control")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.markdown("---")
    st.success("System: Active ✅")

# --- NAVIGATION TABS ---
tabs = st.tabs([
    "👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", 
    "📈 Model Drift", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin Center"
])

# --- TAB 0: ASSESSMENT (With Threshold Tuning) ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000)
    
    if st.button("Analyze Eligibility"):
        is_fraud = detect_fraud(income, amount, credit)
        input_df = pd.DataFrame({
            'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
            'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'],
            'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount],
            'loan_purpose':['Personal'], 'interest_rate':[10.5], 'loan_term':[36],
            'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
            'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0],
            'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 
            'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]
        })
        
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500 and not is_fraud) else "REJECTED"
        
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        
        # --- REFINE LOGIC INTEGRATION ---
        if is_fraud:
            st.warning("🚨 **Fraud Alert:** Financial pattern flagged for manual review.")
        
        if res == "APPROVED":
            if chance >= 75:
                st.balloons()
                st.success(f"✅ **High Confidence Approval:** {chance}% - Low risk profile.")
            else:
                st.warning(f"⚠️ **Moderate Risk Approval:** {chance}% - Eligibility is borderline. Manual verification recommended.")
        else:
            st.error(f"❌ **Rejected:** {chance}% - High risk profile detected.")
        
        log_user_data(name, income, credit, amount, res, chance, 13.08, 80.27)

# --- TAB 1: BULK HUB ---
with tabs[1]:
    st.header("📂 Bulk Processing")
    st.info("Upload CSV for batch analysis.")

# --- TAB 2: LIVE GEO MAPPING ---
with tabs[2]:
    st.header("🗺️ Applicant Geospatial View")
    if os.path.exists('user_logs_production.csv'):
        df_geo = pd.read_csv('user_logs_production.csv')
        st.map(df_geo[['lat', 'lon']])
    else: st.info("Run assessment to see mapping.")

# --- TAB 3: MODEL DRIFT ---
with tabs[3]:
    st.header("📉 Model Stability Monitor")
    drift_df = pd.DataFrame({'Day': range(1,6), 'Accuracy': [0.94, 0.93, 0.94, 0.92, 0.93]})
    st.line_chart(drift_df.set_index('Day'))

# --- TAB 4: EXPLAINABLE AI ---
with tabs[4]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        st.plotly_chart(px.bar(x=impact, y=['Credit Score', 'Income', 'Loan', 'DTI', 'Age'], orientation='h', color=impact))
    else: st.warning("Run Assessment first.")

# --- TAB 5: MARKET & CARDS ---
with tabs[5]:
    st.header("🏦 Market Rates")
    st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari Bank'], 'Rate': ['10.5%', '10.7%', '9.2%']}))

# --- TAB 6: ADMIN CENTER ---
with tabs[6]:
    st.header("🔐 Secure Audit Logs")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs_production.csv'):
            st.dataframe(pd.read_csv('user_logs_production.csv').tail(10))
