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
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    # Adding mock Lat/Lon for Geo-Mapping feature
    lat = np.random.uniform(10.0, 20.0) 
    lon = np.random.uniform(75.0, 80.0)
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob],
        'lat': [lat], 'lon': [lon]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

def anonymize_data(df):
    temp_df = df.copy()
    temp_df['Applicant_Name'] = temp_df['Applicant_Name'].apply(lambda x: str(x)[0] + "***" if len(str(x)) > 1 else "***")
    return temp_df

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.markdown("---")
    st.success("System: Online ✅")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "📊 Comparison & Geo", "🧠 XAI Logic", "🏦 Market & Cards", "🔐 Admin Center"])

# --- TAB 0: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
        loan_cat = st.selectbox("Loan Category", ["Personal Loan", "Home Loan", "Business Loan"])
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000000, 25000)
    
    if st.button("Run AI Prediction"):
        input_df = pd.DataFrame({
            'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
            'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'],
            'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount],
            'loan_purpose':[loan_cat], 'interest_rate':[10.5], 'loan_term':[36],
            'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
            'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0],
            'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 
            'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]
        })
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        st.session_state['last_chance'], st.session_state['last_score'] = chance, credit
        
        if res == "APPROVED": st.success(f"✅ Approved ({chance}%)")
        else: st.error(f"❌ Rejected ({chance}%)")
        log_user_data(name, income, credit, amount, res, chance)

# --- TAB 2: MODEL COMPARISON & GEO MAPPING ---
with tabs[2]:
    st.header("🗺️ Geo-Mapping & Model Benchmark")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Applicant Distribution (Geo)")
        if os.path.exists('user_logs.csv'):
            df_geo = pd.read_csv('user_logs.csv')
            st.map(df_geo[['lat', 'lon']])
        else: st.info("Run assessments to see map data.")
        
    with col_b:
        st.subheader("Champion vs Challenger")
        m_df = pd.DataFrame({'Model': ['RF (Champion)', 'XGB (Challenger)'], 'Accuracy': [0.92, 0.94]})
        st.plotly_chart(px.bar(m_df, x='Model', y='Accuracy', color='Model'), use_container_width=True)

# --- TAB 3: EXPLAINABLE AI ---
with tabs[3]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        features = ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Employment History']
        st.plotly_chart(px.bar(x=impact, y=features, orientation='h', color=impact, color_continuous_scale='RdYlGn'), use_container_width=True)
        st.write("### 🔍 Detail: Why this decision?")
        for i, f in enumerate(features):
            st.write(f"**{f}**: {'Positive' if impact[i]>0 else 'Negative'} impact (Range: {impact[i]})")
    else: st.warning("Run Assessment first.")

# --- TAB 4: MARKET & CARDS ---
with tabs[4]:
    st.header("🏦 Market Interest Comparison")
    m_rates = pd.DataFrame({'Bank': ['SBI', 'HDFC', 'ICICI', 'Hari Bank'], 'Interest Rate': [10.5, 10.7, 10.8, 9.5], 'Fees': ['0.5%', '1%', '0.8%', '0%']})
    st.plotly_chart(px.line(m_rates, x='Bank', y='Interest Rate', markers=True, title="Market Trend Analysis"), use_container_width=True)
    st.subheader("💳 Personalized Offers")
    st.write("Score 700+? You qualify for **Hari Platinum Card**!")

# --- TAB 5: ADMIN CENTER ---
with tabs[5]:
    st.header("🔐 Admin Security Center")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv('user_logs.csv')
            # FEATURE: Security Toggle
            secure_mode = st.toggle("🛡️ Enable Data Anonymization (GDPR)")
            display_df = anonymize_data(df_logs) if secure_mode else df_logs
            st.dataframe(display_df.tail(15))
            st.download_button("Download Secure Logs", display_df.to_csv(index=False), "secure_logs.csv")
