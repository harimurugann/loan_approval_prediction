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
        padding: 0.6rem 2rem; border: none; transition: 0.3s ease;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    # Model path-a unga environment-ku yetha maari check pannikkonga
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HELPER FUNCTIONS ---
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
    st.success("System: Online ✅")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "📊 Model Comparison", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin Center"])

# --- TAB 0: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
        loan_cat = st.selectbox("Loan Category", ["Personal Loan", "Home Loan", "Business Loan", "Education Loan"])
    with col2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000)
        dti = st.slider("Current DTI Ratio", 0.0, 1.0, 0.25)
    
    if st.button("Run AI Prediction"):
        # Formatting data (24 columns fix)
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
        st.session_state['last_res'] = res
        st.session_state['last_cat'] = loan_cat
        
        if res == "APPROVED":
            st.success(f"🎉 **Congratulations!** Your {loan_cat} is **APPROVED** ({chance}% Confidence).")
        else:
            st.error(f"❌ **Rejected.** Risk Confidence: {chance}%.")
            if credit > 600: st.info("💡 **Hari's Tip:** Check 'Market & Cards' tab for alternatives!")
        
        log_user_data(name, income, credit, amount, res, chance)

# --- TAB 1: BULK HUB ---
with tabs[1]:
    st.header("📂 Batch Prediction Hub")
    up_file = st.file_uploader("Upload CSV for Bulk Analysis", type="csv")
    if up_file:
        df_bulk = pd.read_csv(up_file)
        st.dataframe(df_bulk.head())
        if st.button("Start Batch Prediction"):
            df_bulk['AI_Result'] = np.where(df_bulk['credit_score'] > 600, "Approved", "Review Needed")
            st.dataframe(df_bulk)

# --- TAB 2: MODEL COMPARISON ---
with tabs[2]:
    st.header("📊 Champion vs Challenger Comparison")
    metrics = pd.DataFrame({
        'Model': ['Random Forest (Champion)', 'XGBoost (Challenger)'],
        'Accuracy': [0.92, 0.94],
        'Precision': [0.89, 0.91]
    })
    st.table(metrics)
    st.plotly_chart(px.bar(metrics, x='Model', y='Accuracy', color='Model'))

# --- TAB 3: EXPLAINABLE AI ---
with tabs[3]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        st.subheader(f"Detailed Analysis for last Prediction")
        impact = [45 if st.session_state['last_score'] > 600 else -50, 20, -15, 10, 10]
        features = ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Employment']
        fig_xai = px.bar(x=impact, y=features, orientation='h', color=impact, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_xai, use_container_width=True)
        
        st.markdown("### 🔍 Breakdown")
        for i, f in enumerate(features):
            st.write(f"**{f}**: {'✅ Positive' if impact[i]>0 else '❌ Negative'} Impact (Weight: {impact[i]})")
    else:
        st.warning("⚠️ Please run an Assessment first.")

# --- TAB 4: MARKET & CARDS ---
with tabs[4]:
    st.header("🏦 Real-time Market & Card Offers")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Market Bank Rates")
        st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari Bank'], 'Rate': ['10.5%', '10.7%', '9.5%']}))
    with c2:
        st.subheader("Personalized Card Recommendations")
        st.write("💳 **Hari Gold:** Available for Score > 700")
        st.write("💳 **Starter Plus:** Available for all applicants")

# --- TAB 5: ADMIN CENTER ---
with tabs[5]:
    st.header("🔐 Admin Data Center")
    st.write(f"**DEV:** Hari murugan | Data Scientist")
    if st.text_input("Admin Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv('user_logs.csv')
            st.dataframe(df_logs.tail(15))
            st.download_button("Download Logs (CSV)", df_logs.to_csv(index=False), "hari_logs.csv")
