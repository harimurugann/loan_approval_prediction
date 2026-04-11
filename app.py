import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF # PDF generate panna
import base64

# Page Config
st.set_page_config(page_title="Loan Analytics Pro", layout="wide")

# 1. Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- PDF Generation Function ---
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
    pdf.cell(200, 10, txt=f"Monthly Disposable Income: ${income}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="Note: This is an AI-generated report based on provided financial metrics. Please consult a bank official for final confirmation.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- Tabs for Single and Bulk ---
tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Bulk Prediction (CSV)"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.header("Individual Risk Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Applicant Name", "User 1")
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        annual_income = st.number_input("Annual Income ($)", 0, value=50000)
        monthly_income = annual_income / 12

    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        loan_amount = st.number_input("Loan Amount Requested ($)", 0, value=15000)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
        debt_to_income_ratio = st.number_input("DTI Ratio", 0.0, 1.0, 0.1)
        loan_purpose = st.selectbox("Purpose", ["Home", "Car", "Education", "Business", "Medical", "Debt consolidation"])

    # Feature Engineering
    monthly_debt = monthly_income * debt_to_income_ratio
    disposable_income = monthly_income - monthly_debt
    lti = loan_amount / (annual_income if annual_income > 0 else 1)

    input_df = pd.DataFrame({
        'age':[age], 'gender':[gender], 'marital_status':[marital_status], 'education_level':[education],
        'annual_income':[annual_income], 'monthly_income':[monthly_income], 'employment_status':['Employed'],
        'debt_to_income_ratio':[debt_to_income_ratio], 'credit_score':[credit_score], 'loan_amount':[loan_amount],
        'loan_purpose':[loan_purpose], 'interest_rate':[10.5], 'loan_term':[loan_term],
        'installment':[loan_amount/loan_term], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
        'total_credit_limit':[annual_income*1.5], 'current_balance':[loan_amount*0.5], 'delinquency_history':[0],
        'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[monthly_debt], 
        'disposable_income':[disposable_income], 'loan_to_income_ratio':[lti]
    })

    if st.button("Predict & Generate Report"):
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res_text = "APPROVED" if chance >= 50 else "REJECTED"
        
        if chance >= 70: st.success(f"Chance: {chance}% - Likely Approved")
        elif chance >= 40: st.warning(f"Chance: {chance}% - Moderate Risk")
        else: st.error(f"Chance: {chance}% - Likely Rejected")
        
        # PDF Download Button
        pdf_data = create_pdf(name, res_text, chance, round(disposable_income,2), round(monthly_debt,2))
        st.download_button(label="📥 Download PDF Report", data=pdf_data, file_name=f"{name}_Loan_Report.pdf", mime="application/pdf")

# --- TAB 2: BULK PREDICTION ---
with tab2:
    st.header("Bulk Processing")
    st.write("Upload a CSV file with applicant details for mass prediction.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Inga namma feature engineering-a bulk data-vukkum apply pannanum
        data['monthly_income'] = data['annual_income'] / 12
        data['monthly_debt'] = data['monthly_income'] * data['debt_to_income_ratio']
        data['disposable_income'] = data['monthly_income'] - data['monthly_debt']
        data['loan_to_income_ratio'] = data['loan_amount'] / data['annual_income']
        
        # Prediction
        preds = model.predict(data) # Make sure CSV columns match exactly
        data['Loan_Status'] = ["Approved" if p == 1 else "Rejected" for p in preds]
        
        st.write("### Preview of Results:")
        st.dataframe(data.head())
        
        # Download Results
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Result CSV", csv, "bulk_predictions.csv", "text/csv")
