import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import io

# 1. Page Configuration & UI Styling
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;
        padding: 0.6rem 2rem; border: none; transition: 0.3s ease;
    }
    div.stButton > button:first-child:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(0,204,150,0.4); }
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
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

def generate_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="System Audit Logs - Loan Intelligence AI", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    cols = ["Date", "Name", "Income", "Score", "Result"]
    for col in cols: pdf.cell(38, 10, col, 1, 0, 'C')
    pdf.ln()
    for _, row in df.tail(15).iterrows():
        pdf.cell(38, 10, str(row['Timestamp'])[:10], 1)
        pdf.cell(38, 10, str(row['Applicant_Name'])[:15], 1)
        pdf.cell(38, 10, str(row['Annual_Income']), 1)
        pdf.cell(38, 10, str(row['Credit_Score']), 1)
        pdf.cell(38, 10, str(row['Prediction']), 1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR: CHATBOT & BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    chat_input = st.text_input("Ask Bot...", placeholder="e.g. How to fix low credit?")
    if chat_input:
        q = chat_input.lower()
        if "credit" in q: st.info("💡 **Bot:** Pay bills 3 days early and keep utilization below 30%.")
        elif "dti" in q: st.info("💡 **Bot:** Consolidate high-interest debts to lower your monthly DTI.")
        else: st.write("Try asking about 'Credit' or 'Market Rates'.")
    st.markdown("---")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Processing", "📈 Explainable AI", "🏦 Market Rates", "🔐 Admin Center"])

# --- TAB 1: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
        loan_cat = st.selectbox("Loan Category", ["Personal Loan", "Home Loan", "Car Loan", "Business Loan"])
    with col2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000)
        dti = st.slider("Current DTI Ratio", 0.0, 1.0, 0.25)
    
    if st.button("Run AI Prediction"):
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
            if chance >= 75: st.balloons(); st.success(f"🎉 **Congratulations!** Approved ({chance}%)")
            else: st.warning(f"⚠️ **Approved (Moderate Risk):** Confidence: {chance}%")
        else: st.error(f"❌ **Rejected:** Confidence: {chance}%")
        log_user_data(name, income, credit, amount, res, chance)

# --- TAB 2: BULK PROCESSING ---
with tabs[1]:
    st.header("📂 Bulk Assessment Hub")
    up_file = st.file_uploader("Upload CSV", type="csv")
    if up_file:
        df_bulk = pd.read_csv(up_file)
        st.dataframe(df_bulk.head())
        if st.button("Start Batch Prediction"):
            df_bulk['Decision'] = np.where(df_bulk['credit_score'] > 600, "Approved", "Review")
            st.dataframe(df_bulk)

# --- TAB 3: EXPLAINABLE AI ---
with tabs[2]:
    st.header("🧠 Explainable AI (XAI)")
    if 'last_chance' in st.session_state:
        st.subheader(f"Logic for {st.session_state['last_cat']}")
        impact = [45 if st.session_state['last_score'] > 600 else -50, 20, -15, 10, 10]
        features = ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'History']
        fig_xai = px.bar(x=impact, y=features, orientation='h', color=impact, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_xai, use_container_width=True)
        for i, feat in enumerate(features):
            st.write(f"{'✅' if impact[i]>0 else '❌'} **{feat}**: Impact value {impact[i]}")
    else: st.warning("⚠️ Please run Assessment first.")

# --- TAB 4: MARKET RATES ---
with tabs[3]:
    st.header("🏦 Real-time Market Comparison")
    market_data = pd.DataFrame({'Bank': ['SBI', 'HDFC', 'ICICI', 'Hari Bank (AI)'], 'Rate (%)': [10.5, 10.7, 10.8, 9.5], 'Fee': ['0.5%', '1%', '0.8%', '0%']})
    st.table(market_data)
    st.plotly_chart(px.bar(market_data, x='Bank', y='Rate (%)', color='Bank'), use_container_width=True)

# --- TAB 5: ADMIN CENTER ---
with tabs[4]:
    st.header("🔐 Admin Data Center")
    st.write("**DEV:** Hari murugan | Data Scientist")
    if st.text_input("Admin Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv('user_logs.csv')
            st.dataframe(df_logs.tail(10))
            fmt = st.selectbox("Export Format", ["CSV", "PDF"])
            if st.download_button(f"Download {fmt}", df_logs.to_csv().encode('utf-8') if fmt=="CSV" else generate_pdf_report(df_logs), f"Logs.{fmt.lower()}"): pass
