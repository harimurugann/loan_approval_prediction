import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import datetime
import os

# 1. Page Configuration
st.set_page_config(page_title="Loan Intelligence Hub | Dark Edition", layout="wide")

# --- FORCE DARK MODE THEME (CSS) ---
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        /* Input box matrum labels-ku dark theme support */
        label { color: #ffffff !important; }
        .stTextInput>div>div>input { color: white; }
        .stNumberInput>div>div>input { color: white; }
        /* Sidebar dark theme */
        [data-testid="stSidebar"] {
            background-color: #161b22;
        }
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
        'Applicant_Name': [name],
        'Annual_Income': [income],
        'Credit_Score': [credit],
        'Loan_Amount': [amount],
        'Prediction': [result],
        'Probability_%': [prob]
    })
    if not os.path.isfile(log_file):
        log_entry.to_csv(log_file, index=False)
    else:
        log_entry.to_csv(log_file, mode='a', header=False, index=False)

# Session State for Security
if 'admin_pwd' not in st.session_state:
    st.session_state['admin_pwd'] = "admin123"

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["👤 Assessment", "📂 Bulk Upload", "📈 Insights", "🔐 Admin Panel"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tab1:
    st.header("Single Applicant Risk Check")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Applicant Name", "Guest User")
        age = st.number_input("Age", 18, 100, 30)
        annual_income = st.number_input("Annual Income ($)", 0, value=50000)
        
    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        loan_amount = st.number_input("Loan Amount ($)", 0, value=15000)
        dti = st.number_input("DTI Ratio", 0.0, 1.0, 0.15)

    # Dynamic Tips (Dark Mode-ku thagundha warnings)
    st.markdown("---")
    if credit_score < 600:
        st.warning("⚠️ **Low Credit Score:** Risk of rejection is high.")
    if dti > 0.45:
        st.error("🚨 **High Debt Warning:** Debt-to-Income ratio is over the safety limit.")

    # Data for prediction
    input_df = pd.DataFrame({
        'age':[age], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
        'annual_income':[annual_income], 'monthly_income':[annual_income/12], 'employment_status':['Employed'],
        'debt_to_income_ratio':[dti], 'credit_score':[credit_score], 'loan_amount':[loan_amount],
        'loan_purpose':['Business'], 'interest_rate':[10.5], 'loan_term':[36],
        'installment':[loan_amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
        'total_credit_limit':[annual_income*1.5], 'current_balance':[loan_amount*0.5], 'delinquency_history':[0],
        'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[(annual_income/12)*dti], 
        'disposable_income':[(annual_income/12) - ((annual_income/12)*dti)], 'loan_to_income_ratio':[loan_amount/annual_income if annual_income > 0 else 0]
    })

    if st.button("Run Prediction"):
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        st.session_state['last_chance'] = chance
        res = "APPROVED" if chance >= 50 else "REJECTED"
        
        log_user_data(name, annual_income, credit_score, loan_amount, res, chance)
        st.success(f"Final Decision: {res} ({chance}% Confidence)")

# --- TAB 3: INSIGHTS ---
with tab3:
    st.header("Decision Insights")
    cur_chance = st.session_state.get('last_chance', 0)
    color = "#00CC96" if cur_chance >= 70 else ("#FFAA00" if cur_chance >= 40 else "#FF4B4B")
    
    st.subheader(f"Current Probability Score: :{color}[{cur_chance}%]")
    
    # Feature Importance Visualization
    imp_df = pd.DataFrame({'Feature': ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Age'], 'Impact %': [45, 25, 15, 10, 5]})
    fig = px.bar(imp_df, x='Impact %', y='Feature', orientation='h', template="plotly_dark")
    fig.update_traces(marker_color=color)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: ADMIN PANEL ---
with tab4:
    st.header("🔐 Admin Security")
    pwd_input = st.text_input("Admin Password", type="password")
    
    if pwd_input == st.session_state['admin_pwd']:
        st.success("Access Granted.")
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv(    'user_logs.csv')
            st.dataframe(df_logs, use_container_width=True)
            if st.button("Delete Logs"):
                os.remove('user_logs.csv')
                st.rerun()
    elif pwd_input != "":
        st.error("Access Denied.")

with st.sidebar:
    st.title("🛡️ System Info")
    st.write("App Status: Active")
    st.write("Theme: Permanent Dark Mode")
    st.info("AI Data Engineer Dashboard v2.0")
