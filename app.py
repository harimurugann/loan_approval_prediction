import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from fpdf import FPDF
import datetime
import os

# 1. Page Configuration
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

# UI Branding
st.markdown("""<style>.stApp {background-color: #0e1117; color: white;} div.stButton > button:first-child {background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;}</style>""", unsafe_allow_html=True)

# 2. Model Loading
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HELPERS ---
def generate_pdf(df, title):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"HARI AI BANK - {title.upper()}", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Date: {datetime.date.today()}", 0, 1, 'R')
    pdf.ln(10)
    pdf.set_fill_color(0, 204, 150); pdf.set_text_color(255, 255, 255)
    cols = ['Date', 'Client', 'Income', 'Score', 'Status']
    w = [30, 50, 35, 35, 40]
    for i, col in enumerate(cols): pdf.cell(w[i], 10, col, 1, 0, 'C', 1)
    pdf.ln(); pdf.set_text_color(0, 0, 0)
    for _, row in df.iterrows():
        pdf.cell(30, 10, str(row['Timestamp'])[:10], 1); pdf.cell(50, 10, str(row['Applicant_Name'])[:18], 1)
        pdf.cell(35, 10, f"${row['Annual_Income']}", 1); pdf.cell(35, 10, str(row['Credit_Score']), 1)
        pdf.cell(40, 10, str(row['Prediction']), 1); pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

def log_data(name, income, credit, amount, result, prob):
    file = 'user_logs_pro.csv'
    log = pd.DataFrame({'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M')], 'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit], 'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob], 'lat': [13.0827], 'lon': [80.2707]})
    if not os.path.isfile(file): log.to_csv(file, index=False)
    else: log.to_csv(file, mode='a', header=False, index=False)

# --- TABS (Strict Indexing 0-6) ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", "📉 Model Drift", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin CRM Hub"])

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
        prob = model.predict_proba(input_df)[0][1]; chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        st.session_state['last_score'] = credit
        if res == "APPROVED": st.success(f"✅ Approved ({chance}%)")
        else: st.error(f"❌ Rejected ({chance}%)")
        log_data(u_name, income, credit, amount, res, chance)

# --- TABS 1-5 Content ---
with tabs[1]: st.header("📂 Bulk Processing"); up = st.file_uploader("Upload CSV", type="csv")
with tabs[2]: st.header("🗺️ Geospatial View"); df_g = pd.read_csv('user_logs_pro.csv') if os.path.exists('user_logs_pro.csv') else None; st.map(df_g[['lat', 'lon']]) if df_g is not None else st.info("No data.")
with tabs[3]: st.header("📉 Performance Drift"); st.line_chart([0.94, 0.92, 0.95, 0.91, 0.93])
with tabs[4]: st.header("🧠 Decision Logic"); st.info("Run Assessment to see bars.")
with tabs[5]: st.header("🏦 Market Rates"); st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari AI'], 'Rate': ['10.5%', '10.7%', '9.2%']}))

# --- TAB 6: ADMIN CRM (FIXED LOGIC) ---
with tabs[6]:
    st.header("🔐 Admin CRM & Audit Center")
    pwd = st.text_input("Access Key", type="password")
    if pwd == "admin123":
        if os.path.exists('user_logs_pro.csv'):
            df = pd.read_csv('user_logs_pro.csv')
            search = st.text_input("🔍 Search Client Name")
            filtered = df[df['Applicant_Name'].str.contains(search, case=False, na=False)] if search else df
            
            st.subheader("Selection & Batch Download")
            selected = st.multiselect("Select Clients", options=filtered['Applicant_Name'].unique())
            final_df = filtered[filtered['Applicant_Name'].isin(selected)] if selected else filtered
            
            st.dataframe(final_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1: st.download_button("📥 Export CSV", final_df.to_csv(index=False), "Bank_Logs.csv", "text/csv")
            with col2: st.download_button("📄 Export PDF", generate_pdf(final_df, "Audit Report"), "Bank_Audit.pdf", "application/pdf")
        else: st.warning("No logs found.")
    elif pwd != "": st.error("Wrong Password.")
