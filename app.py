import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import datetime
import os

# 1. Page Configuration
st.set_page_config(page_title="Loan Intelligence Hub", layout="wide")

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

# --- PASSWORD MANAGEMENT (Session Based) ---
if 'admin_pwd' not in st.session_state:
    st.session_state['admin_pwd'] = "admin123" # Initial Default Password

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["👤 Assessment", "📂 Bulk", "📈 Analytics", "🔐 Admin"])

# --- TAB 1: INDIVIDUAL ---
with tab1:
    st.header("Individual Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "Guest User")
        annual_income = st.number_input("Annual Income ($)", 0, value=50000)
    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        loan_amount = st.number_input("Loan Amount ($)", 0, value=15000)
        dti = st.number_input("DTI Ratio", 0.0, 1.0, 0.1)

    # Dummy Feature engineering for the model
    input_df = pd.DataFrame({
        'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
        'annual_income':[annual_income], 'monthly_income':[annual_income/12], 'employment_status':['Employed'],
        'debt_to_income_ratio':[dti], 'credit_score':[credit_score], 'loan_amount':[loan_amount],
        'loan_purpose':['Business'], 'interest_rate':[10.5], 'loan_term':[36],
        'installment':[loan_amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
        'total_credit_limit':[annual_income*1.5], 'current_balance':[loan_amount*0.5], 'delinquency_history':[0],
        'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[(annual_income/12)*dti], 
        'disposable_income':[(annual_income/12) - ((annual_income/12)*dti)], 'loan_to_income_ratio':[loan_amount/annual_income if annual_income > 0 else 0]
    })

    if st.button("Analyze"):
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit_score
        res = "APPROVED" if chance >= 50 else "REJECTED"
        log_user_data(name, annual_income, credit_score, loan_amount, res, chance)
        st.success(f"Result: {res} ({chance}%)")

# --- TAB 4: SECURED ADMIN & PWD CHANGE ---
with tab4:
    st.header("🔐 Admin Access")
    
    # 1. Login Gate
    pwd_input = st.text_input("Enter Admin Password", type="password")
    
    if pwd_input == st.session_state['admin_pwd']:
        st.success("Welcome Admin")
        
        # --- PASSWORD CHANGE OPTION ---
        with st.expander("🛠️ Settings: Change Admin Password"):
            new_pwd = st.text_input("New Password", type="password")
            confirm_pwd = st.text_input("Confirm New Password", type="password")
            if st.button("Update Password"):
                if new_pwd == confirm_pwd and new_pwd != "":
                    st.session_state['admin_pwd'] = new_pwd
                    st.success("Password updated for this session! Use the new password next time.")
                else:
                    st.error("Passwords do not match!")

        # --- VIEW LOGS ---
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv('user_logs.csv')
            st.subheader("System Logs")
            st.dataframe(df_logs, use_container_width=True)
            
            if st.button("🗑️ Reset Logs"):
                os.remove('user_logs.csv')
                st.rerun()
        else:
            st.info("No logs found.")
            
    elif pwd_input != "":
        st.error("Invalid Password")
