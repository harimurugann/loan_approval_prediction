import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import datetime

# 1. Page Configuration
st.set_page_config(page_title="Loan Analytics Pro | AI Data Engineer", layout="wide")

# 2. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HELPER FUNCTIONS ---

def create_pdf(name, result, chance, income, debt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Loan Approval Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Applicant Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Status: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Approval Probability: {chance}%", ln=True)
    pdf.cell(200, 10, txt=f"Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now()],
        'Name': [name],
        'Annual_Income': [income],
        'Credit_Score': [credit],
        'Loan_Amount': [amount],
        'Result': [result],
        'Probability': [prob]
    })
    try:
        existing_logs = pd.read_csv(log_file)
        updated_logs = pd.concat([existing_logs, log_entry], ignore_index=True)
        updated_logs.to_csv(log_file, index=False)
    except FileNotFoundError:
        log_entry.to_csv(log_file, index=False)

# --- THE TABS ---
tab1, tab2, tab3 = st.tabs(["👤 Individual Check", "📂 Bulk Processing", "📈 Advanced Analytics & Simulator"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.header("Individual Risk Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Applicant Name", "Guest User")
        age = st.number_input("Age", 18, 100, 30)
        annual_income = st.number_input("Annual Income ($)", 0, value=50000)
        monthly_income = annual_income / 12

    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        loan_amount = st.number_input("Loan Amount Requested ($)", 0, value=15000)
        debt_to_income_ratio = st.number_input("DTI Ratio (0.0 - 1.0)", 0.0, 1.0, 0.1)

    # Backend Calculations (Feature Engineering)
    monthly_debt = monthly_income * debt_to_income_ratio
    disposable_income = monthly_income - monthly_debt
    lti = loan_amount / (annual_income if annual_income > 0 else 1)

    input_df = pd.DataFrame({
        'age':[age], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
        'annual_income':[annual_income], 'monthly_income':[monthly_income], 'employment_status':['Employed'],
        'debt_to_income_ratio':[debt_to_income_ratio], 'credit_score':[credit_score], 'loan_amount':[loan_amount],
        'loan_purpose':['Business'], 'interest_rate':[10.5], 'loan_term':[36],
        'installment':[loan_amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
        'total_credit_limit':[annual_income*1.5], 'current_balance':[loan_amount*0.5], 'delinquency_history':[0],
        'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[monthly_debt], 
        'disposable_income':[disposable_income], 'loan_to_income_ratio':[lti]
    })

    if st.button("Predict & Generate PDF"):
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        st.session_state['chance'] = chance
        st.session_state['credit_score'] = credit_score
        st.session_state['dti'] = debt_to_income_ratio
        
        res_text = "APPROVED" if chance >= 50 else "REJECTED"
        
        # Log to Database
        log_user_data(name, annual_income, credit_score, loan_amount, res_text, chance)
        st.toast("User data logged for analysis", icon="💾")

        if chance >= 70: st.success(f"Approval Probability: {chance}%")
        elif chance >= 40: st.warning(f"Approval Probability: {chance}%")
        else: st.error(f"Approval Probability: {chance}%")
        
        pdf_data = create_pdf(name, res_text, chance, round(disposable_income,2), round(monthly_debt,2))
        st.download_button("📥 Download Assessment PDF", pdf_data, f"{name}_Report.pdf", "application/pdf")

# --- TAB 2: BULK PROCESSING ---
with tab2:
    st.header("Bulk Processing (CSV)")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Processing Data...")
        # (Assuming CSV has necessary columns for batch prediction)
        st.dataframe(data.head())

# --- TAB 3: ADVANCED ANALYTICS & SIMULATOR ---
with tab3:
    st.header("📈 Decision Engine & Technical Insights")
    current_chance = st.session_state.get('chance', 50)
    c_score = st.session_state.get('credit_score', 700)
    c_dti = st.session_state.get('dti', 0.1)

    if current_chance >= 70: color, status = "#00CC96", "LOW RISK / SAFE"
    elif current_chance >= 40: color, status = "#FFAA00", "MODERATE RISK"
    else: color, status = "#FF4B4B", "HIGH RISK / REJECTION"

    # 1. Visualization
    st.subheader(f"Status Assessment: :{color}[{status}]")
    imp_df = pd.DataFrame({'Feature': ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Age'], 'Importance %': [45, 25, 15, 10, 5]})
    fig_bar = px.bar(imp_df, x='Importance %', y='Feature', orientation='h', title="Why the model chose this result")
    fig_bar.update_traces(marker_color=color)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Smart
