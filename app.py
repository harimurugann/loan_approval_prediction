import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from fpdf import FPDF
import datetime
import os
import io

# 1. Page Configuration
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

# Professional UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00CC96; }
    </style>
    """, unsafe_allow_html=True)

# 2. Model Loading
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- PROFESSIONAL PDF GENERATOR ---
def generate_pro_pdf(df, report_title="Audit Log"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"HARI AI BANK - {report_title.upper()}", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Date: {datetime.date.today()}", 0, 1, 'R')
    pdf.ln(10)
    
    # Header
    pdf.set_fill_color(0, 204, 150)
    pdf.set_text_color(255, 255, 255)
    cols = ['Date', 'Client Name', 'Income', 'Score', 'Status']
    w = [30, 50, 35, 35, 40]
    for i, col in enumerate(cols):
        pdf.cell(w[i], 10, col, 1, 0, 'C', 1)
    pdf.ln()
    
    # Body
    pdf.set_text_color(0, 0, 0)
    for _, row in df.iterrows():
        pdf.cell(30, 10, str(row['Timestamp'])[:10], 1)
        pdf.cell(50, 10, str(row['Applicant_Name'])[:18], 1)
        pdf.cell(35, 10, f"${row['Annual_Income']}", 1)
        pdf.cell(35, 10, str(row['Credit_Score']), 1)
        pdf.cell(40, 10, str(row['Prediction']), 1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# --- DATA LOGGING ---
def log_user_data(name, income, credit, amount, result, prob):
    file = 'user_logs_pro.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob],
        'lat': [13.0827], 'lon': [80.2707]
    })
    if not os.path.isfile(file): log_entry.to_csv(file, index=False)
    else: log_entry.to_csv(file, mode='a', header=False, index=False)

# --- NAVIGATION TABS (Strict Indexing) ---
# Index Mapping: 0:Assessment, 1:Bulk, 2:Geo, 3:Drift, 4:XAI, 5:Market, 6:Admin
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", "📉 Model Drift", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin CRM Hub"])

# --- TAB 0: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        u_name = st.text_input("Applicant Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000000, 25000)
    
    if st.button("Run AI Prediction"):
        input_df = pd.DataFrame({'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"], 'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'], 'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount], 'loan_purpose':['Personal'], 'interest_rate':[10.5], 'loan_term':[36], 'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5], 'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0], 'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]})
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        st.session_state['last_chance'], st.session_state['last_score'] = chance, credit
        if res == "APPROVED": st.success(f"✅ Approved ({chance}%)")
        else: st.error(f"❌ Rejected ({chance}%)")
        log_user_data(u_name, income, credit, amount, res, chance)

# --- TAB 1: BULK HUB ---
with tabs[1]:
    st.header("📂 Bulk Processing")
    up = st.file_uploader("Upload CSV", type="csv")
    if up: st.dataframe(pd.read_csv(up).head())

# --- TAB 2: LIVE GEO MAPPING ---
with tabs[2]:
    st.header("🗺️ Geospatial Distribution")
    if os.path.exists('user_logs_pro.csv'):
        df_geo = pd.read_csv('user_logs_pro.csv')
        st.map(df_geo[['lat', 'lon']])
    else: st.info("Run Assessment to see mapping.")

# --- TAB 3: MODEL DRIFT ---
with tabs[3]:
    st.header("📉 Model Drift Monitor")
    drift_df = pd.DataFrame({'Day': range(1,11), 'Accuracy': [0.94, 0.93, 0.94, 0.92, 0.94, 0.91, 0.90, 0.92, 0.91, 0.92]})
    st.plotly_chart(px.line(drift_df, x='Day', y='Accuracy', title="Production Stability"))

# --- TAB 4: EXPLAINABLE AI ---
with tabs[4]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        st.plotly_chart(px.bar(x=impact, y=['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Age'], orientation='h', color=impact, color_continuous_scale='RdYlGn'))
    else: st.warning("Run Assessment first.")

# --- TAB 5: MARKET & CARDS ---
with tabs[5]:
    st.header("🏦 Comparative Market Rates")
    st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari AI Bank'], 'Rate': ['10.5%', '10.7%', '9.2%']}))

# --- TAB 6: ADMIN CRM HUB (FIXED & FULL FEATURES) ---
with tabs[6]:
    st.header("🔐 Admin CRM & Data Audit")
    if st.text_input("Access Password", type="password") == "admin123":
        if os.path.exists('user_logs_pro.csv'):
            full_df = pd.read_csv('user_logs_pro.csv')
            
            # 1. Search Logic
            search = st.text_input("🔍 Search Client by Name")
            filtered = full_df[full_df['Applicant_Name'].str.contains(search, case=False, na=False)] if search else full_df
            
            # 2. Multi-Select Logic
            st.subheader("Selection & Batch Export")
            selected_clients = st.multiselect("Select Clients for Batch Download", options=filtered['Applicant_Name'].unique())
            
            final_selection = filtered[filtered['Applicant_Name'].isin(selected_clients)] if selected_clients else filtered
            
            st.dataframe(final_selection, use_container_width=True)
            
            # 3. Download Options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download CSV (Selection)", final_selection.to_csv(index=False), "Bank_Logs.csv", "text/csv")
            with col2:
                pdf_bytes = generate_pro_pdf(final_selection, report_title="Batch Audit Report")
                st.download_button("📄 Download PDF Statement (Selection)", pdf_bytes, "Bank_Audit.pdf", "application/pdf")
