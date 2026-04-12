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

def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- SIDEBAR BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.markdown("---")
    st.info("💡 **Tip:** Hari says higher income with lower DTI guarantees 90% approval.")

# --- NAVIGATION TABS ---
# Corrected Index Order: 0:Assessment, 1:Bulk, 2:Comparison, 3:XAI, 4:Market, 5:Admin
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "📊 Model Comparison", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin Center"])

# --- TAB 0: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name", "Guest")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
        loan_cat = st.selectbox("Loan Category", ["Personal Loan", "Home Loan", "Business Loan"])
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000)
    
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
        
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        st.session_state['last_res'] = res
        
        if res == "APPROVED": st.success(f"🎉 **Congratulations!** {res} ({chance}%)")
        else:
            st.error(f"❌ **Rejected.** Chance: {chance}%")
            if credit > 600: st.info("🏆 **Hari's Suggestion:** You qualify for a Premium Card!")
        
        log_user_data(name, income, credit, amount, res, chance)

# --- TAB 2: MODEL COMPARISON ---
with tabs[2]:
    st.header("📊 Champion vs Challenger Benchmarking")
    m_df = pd.DataFrame({'Model': ['RF (Hari Champion)', 'XGB (Challenger)'], 'Accuracy': [0.92, 0.94], 'F1': [0.91, 0.93]})
    st.plotly_chart(px.bar(m_df, x='Model', y='Accuracy', color='Model'), use_container_width=True)

# --- TAB 3: EXPLAINABLE AI (FIXED DESCRIPTION) ---
with tabs[3]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        # Visual Chart
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        features = ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Employment']
        st.plotly_chart(px.bar(x=impact, y=features, orientation='h', color=impact, color_continuous_scale='RdYlGn'), use_container_width=True)
        
        # FIXED: Description below chart
        st.subheader("📝 Decision Breakdown")
        for i, feat in enumerate(features):
            val = impact[i]
            status = "✅ Major Positive" if val > 30 else ("🟢 Minor Positive" if val > 0 else "❌ Negative Impact")
            st.write(f"**{feat}**: {status} (Range: {val})")
    else: st.warning("Run Assessment first.")

# --- TAB 4: MARKET & CARDS (FIXED BLANK) ---
with tabs[4]:
    st.header("🏦 Market Rates & Card Recommendations")
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Top Bank Rates")
        st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari Bank'], 'Rate': ['10.5%', '10.7%', '9.5%']}))
    with c_right:
        st.subheader("Hari's Card Picks")
        st.write("💳 **Gold Card:** Eligible if Score > 700")
        st.write("💳 **Starter Card:** Eligible for all applicants")

# --- TAB 5: ADMIN CENTER (FIXED TOGGLE) ---
with tabs[5]:
    st.header("🔐 Admin Security Center")
    st.write("**DEV:** Hari murugan | Data Scientist")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            df = pd.read_csv('user_logs.csv')
            # FIXED: Toggle is now here
            security_on = st.toggle("🛡️ Enable Data Anonymization (GDPR Mode)")
            if security_on:
                df = anonymize_data(df)
            st.dataframe(df.tail(15))
            st.download_button("Download CSV", df.to_csv(index=False), "hari_murugan_logs.csv")
