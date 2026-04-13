import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from fpdf import FPDF
import datetime
import os
import io

# 1. Page Configuration & Custom Styling
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00CC96; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

model = load_model()

# --- PROFESSIONAL PDF GENERATOR ---
def generate_bank_pdf(df, title="Client Data Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"HARI AI BANK - {title.upper()}", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Report Date: {datetime.date.today()}", 0, 1, 'R')
    pdf.ln(10)

    # Table Header
    pdf.set_fill_color(0, 204, 150)
    pdf.set_text_color(255, 255, 255)
    headers = ['Date', 'Name', 'Income', 'Score', 'Decision']
    widths = [35, 45, 35, 35, 35]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 10, h, 1, 0, 'C', 1)
    pdf.ln()

    # Table Content
    pdf.set_text_color(0, 0, 0)
    for _, row in df.iterrows():
        pdf.cell(35, 10, str(row['Timestamp'])[:10], 1)
        pdf.cell(45, 10, str(row['Applicant_Name'])[:15], 1)
        pdf.cell(35, 10, f"${row['Annual_Income']}", 1)
        pdf.cell(35, 10, str(row['Credit_Score']), 1)
        pdf.cell(35, 10, str(row['Prediction']), 1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin-1')

# --- LOGGING DATA ---
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs_pro.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now()],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob],
        'lat': [13.0827], 'lon': [80.2707]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", "📉 Model Drift", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin CRM Hub"])

# --- TAB 0: ASSESSMENT (Same as before) ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        u_name = st.text_input("Applicant Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Requested Loan ($)", 0, 100000000, 25000)
    
    if st.button("Run AI Prediction"):
        input_df = pd.DataFrame({'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"], 'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'], 'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount], 'loan_purpose':['Personal'], 'interest_rate':[10.5], 'loan_term':[36], 'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5], 'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0], 'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]})
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        if res == "APPROVED": st.success(f"✅ Approved ({chance}%)")
        else: st.error(f"❌ Rejected ({chance}%)")
        log_user_data(u_name, income, credit, amount, res, chance)

# --- TAB 6: ADMIN CRM HUB (NEW ADVANCED FEATURES) ---
with tabs[6]:
    st.header("🔐 Client Data CRM & Audit Hub")
    if st.text_input("Admin Password", type="password") == "admin123":
        if os.path.exists('user_logs_pro.csv'):
            df = pd.read_csv('user_logs_pro.csv')
            
            # Search Feature
            st.subheader("🔍 Search & Filter")
            search_query = st.text_input("Search by Client Name", "")
            
            if search_query:
                filtered_df = df[df['Applicant_Name'].str.contains(search_query, case=False, na=False)]
            else:
                filtered_df = df

            # Multi-Selection Download Feature
            st.subheader("📋 Client Data Management")
            st.write("Select specific clients to download their data as a batch.")
            
            # Displaying data with a multi-select option
            selected_names = st.multiselect("Select Clients", options=filtered_df['Applicant_Name'].unique())
            
            if selected_names:
                final_df = filtered_df[filtered_df['Applicant_Name'].isin(selected_names)]
            else:
                final_df = filtered_df

            st.dataframe(final_df, use_container_width=True)

            # Download Options for Filtered/Selected Data
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Download Selection (CSV)", final_df.to_csv(index=False), "Bank_Selection.csv", "text/csv")
            with c2:
                pdf_data = generate_bank_pdf(final_df, title="Selected Clients Audit")
                st.download_button("📄 Download Selection (PDF)", pdf_data, "Bank_Selection.pdf", "application/pdf")
                
            st.markdown("---")
            st.info("💡 Tip: Use the search bar first, then multi-select clients for an efficient audit report.")
        else:
            st.warning("No data found in logs.")
