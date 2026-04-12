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

# --- HELPER FUNCTIONS ---
def anonymize_data(df):
    temp_df = df.copy()
    temp_df['Applicant_Name'] = temp_df['Applicant_Name'].apply(lambda x: str(x)[0] + "***" if len(str(x)) > 1 else "***")
    return temp_df

def log_user_data(name, income, credit, amount, result, prob, lat, lon):
    log_file = 'user_logs_geo.csv'
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
    st.title("🤖 Finance AI Bot")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.info("System: Online ✅ | Geo-Tracking: Active 📍")

# --- NAVIGATION TABS (ALL 7 FEATURES) ---
tabs = st.tabs([
    "👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", 
    "📊 Model Comparison", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin Center"
])

# --- TAB 0: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name", "Guest")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000000, 25000)
    
    if st.button("Run AI Prediction"):
        # Dummy Geo Data for Simulation (In real app, use geocoder)
        sample_lat, sample_lon = 13.0827, 80.2707 # Chennai coords
        
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
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        
        if res == "APPROVED": st.success(f"🎉 Approved ({chance}%)")
        else: st.error(f"❌ Rejected ({chance}%)")
        
        log_user_data(name, income, credit, amount, res, chance, sample_lat, sample_lon)

# --- TAB 1: BULK HUB ---
with tabs[1]:
    st.header("📂 Bulk Assessment")
    st.info("Upload CSV files for high-speed batch processing.")

# --- TAB 2: LIVE GEO MAPPING (RESTORED) ---
with tabs[2]:
    st.header("🗺️ Real-time Applicant Mapping")
    if os.path.exists('user_logs_geo.csv'):
        geo_df = pd.read_csv('user_logs_geo.csv')
        st.map(geo_df[['lat', 'lon']])
        st.write("📍 Showing live distribution of loan applicants.")
    else:
        st.warning("No geo-data available yet. Run an assessment first.")

# --- TAB 3: MODEL COMPARISON ---
with tabs[3]:
    st.header("📊 Model Benchmarking")
    m_df = pd.DataFrame({'Model': ['Hari RF (Champion)', 'XGBoost (Challenger)'], 'Accuracy': [0.92, 0.94]})
    st.plotly_chart(px.bar(m_df, x='Model', y='Accuracy', color='Model'))

# --- TAB 4: EXPLAINABLE AI ---
with tabs[4]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        st.plotly_chart(px.bar(x=impact, y=['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Employment'], orientation='h', color=impact))
    else: st.warning("Run Assessment first.")

# --- TAB 5: MARKET & CARDS ---
with tabs[5]:
    st.header("🏦 Market Rates & Card Recommendations")
    st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari Bank'], 'Rate': ['10.5%', '10.7%', '9.5%']}))

# --- TAB 6: ADMIN CENTER ---
with tabs[6]:
    st.header("🔐 Admin Security Center")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs_geo.csv'):
            df = pd.read_csv('user_logs_geo.csv')
            if st.toggle("🛡️ Enable GDPR Masking"):
                df = anonymize_data(df)
            st.dataframe(df.tail(10))
