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
def log_data(name, income, credit, amount, res, prob):
    file = 'user_logs_production.csv'
    log = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [res], 'Probability_%': [prob],
        'lat': [13.08], 'lon': [80.27]
    })
    if not os.path.isfile(file): log.to_csv(file, index=False)
    else: log.to_csv(file, mode='a', header=False, index=False)

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", "📈 Model Drift", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin Center", "🚀 Future Roadmap"])

# --- TAB 0: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        u_name = st.text_input("Full Name", "Guest")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000)
    
    if st.button("Analyze Eligibility"):
        input_df = pd.DataFrame({'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"], 'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'], 'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount], 'loan_purpose':['Personal'], 'interest_rate':[10.5], 'loan_term':[36], 'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5], 'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0], 'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]})
        prob = model.predict_proba(input_df)[0][1]; chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        st.session_state['last_chance'], st.session_state['last_score'] = chance, credit
        if res == "APPROVED": st.success(f"✅ Approved ({chance}%)")
        else: st.error(f"❌ Rejected ({chance}%)")
        log_data(u_name, income, credit, amount, res, chance)

# --- TAB 1 to 5 (Simplified for brevity, maintain your previous logic) ---
with tabs[1]: st.header("📂 Bulk Hub"); st.file_uploader("Upload CSV", type="csv")
with tabs[2]: 
    st.header("🗺️ Live Geo Mapping")
    if os.path.exists('user_logs_production.csv'): st.map(pd.read_csv('user_logs_production.csv')[['lat', 'lon']])
with tabs[3]: st.header("📈 Model Drift"); st.line_chart([0.94, 0.92, 0.95, 0.93])
with tabs[4]: st.header("🧠 Explainable AI"); st.info("Run Assessment first.")
with tabs[5]: st.header("🏦 Market & Cards"); st.write("Bank Rates Comparison")

# --- TAB 6: ADMIN CENTER (FIXED & ADDED LOGIC) ---
with tabs[6]:
    st.header("🔐 Admin Security Center")
    st.write("Secure access to applicant data and system logs.")
    
    password = st.text_input("Enter Admin Password", type="password")
    if password == "admin123":
        st.success("Access Granted!")
        if os.path.exists('user_logs_production.csv'):
            logs_df = pd.read_csv('user_logs_production.csv')
            
            # Metrics
            col1, col2 = st.columns(2)
            col1.metric("Total Applications", len(logs_df))
            col2.metric("System Health", "Good")
            
            st.write("### Recent Activity Logs")
            st.dataframe(logs_df.tail(10))
            
            st.download_button(
                label="📥 Download All Logs (CSV)",
                data=logs_df.to_csv(index=False),
                file_name=f"loan_logs_{datetime.datetime.now().date()}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No logs found. Run an assessment to generate data.")
    elif password != "":
        st.error("Incorrect Password. Access Denied.")
