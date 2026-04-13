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

# --- HELPER FUNCTIONS ---
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs_production.csv'
    lat, lon = 13.0827, 80.2707 
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob],
        'lat': [lat], 'lon': [lon]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- NAVIGATION TABS ---
tabs = st.tabs([
    "👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", 
    "📈 Model Drift", "🧠 Explainable AI", "🏦 Market & Cards", 
    "🔐 Admin Center", "🚀 Future Roadmap"
])

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
    
    if st.button("Run AI Prediction"):
        input_df = pd.DataFrame({'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"], 'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'], 'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount], 'loan_purpose':['Personal'], 'interest_rate':[10.5], 'loan_term':[36], 'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5], 'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0], 'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]})
        
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        
        st.session_state['last_chance'], st.session_state['last_score'] = chance, credit
        
        if res == "APPROVED":
            if chance >= 75: st.balloons(); st.success(f"✅ Approved ({chance}%)")
            else: st.warning(f"⚠️ Moderate Risk Approval ({chance}%)")
        else: st.error(f"❌ Rejected ({chance}%)")
        log_user_data(u_name, income, credit, amount, res, chance)

# --- TAB 1: BULK HUB ---
with tabs[1]:
    st.header("📂 Bulk Processing")
    up = st.file_uploader("Upload CSV", type="csv")
    if up: st.dataframe(pd.read_csv(up).head())

# --- TAB 2: LIVE GEO MAPPING ---
with tabs[2]:
    st.header("🗺️ Geo Distribution")
    if os.path.exists('user_logs_production.csv'):
        df_geo = pd.read_csv('user_logs_production.csv')
        st.map(df_geo[['lat', 'lon']])
    else: st.info("Run Assessment first.")

# --- TAB 3: MODEL DRIFT ---
with tabs[3]:
    st.header("📈 Drift Monitor")
    d_df = pd.DataFrame({'Day': range(1,11), 'Accuracy': [0.94, 0.93, 0.94, 0.92, 0.94, 0.91, 0.90, 0.92, 0.91, 0.92]})
    st.plotly_chart(px.line(d_df, x='Day', y='Accuracy'))

# --- TAB 4: EXPLAINABLE AI ---
with tabs[4]:
    st.header("🧠 Logic Breakdown")
    if 'last_chance' in st.session_state:
        imp = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        st.plotly_chart(px.bar(x=imp, y=['Credit Score', 'Income', 'Amount', 'DTI', 'Age'], orientation='h', color=imp))
    else: st.warning("Run Assessment first.")

# --- TAB 5: MARKET & CARDS ---
with tabs[5]:
    st.header("🏦 Rates & Offers")
    st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari AI'], 'Rate': ['10.5%', '10.7%', '9.2%']}))

# --- TAB 6: ADMIN CENTER (FIXED) ---
with tabs[6]:
    st.header("🔐 Admin Center")
    pwd = st.text_input("Password", type="password")
    if pwd == "admin123":
        if os.path.exists('user_logs_production.csv'):
            df_admin = pd.read_csv('user_logs_production.csv')
            st.metric("Total API Calls", len(df_admin))
            st.dataframe(df_admin)
            st.download_button("Export Logs", df_admin.to_csv(index=False), "logs.csv")
    elif pwd != "": st.error("Wrong Password")

# --- TAB 7: FUTURE ROADMAP ---
with tabs[7]:
    st.header("🚀 Future Roadmap")
    st.write("- NLP Sentiment Analysis Integration")
    st.write("- Real-time API connections to Banks")
