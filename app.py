import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved pipeline
# Ensure 'loan_model_pipeline.sav' is in the same GitHub folder
try:
    model = joblib.load('loan_model_pipeline.sav')
except Exception as e:
    st.error(f"Error loading model: {e}")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant details below to check loan eligibility.")

# Create input fields based on your dataset columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    annual_income = st.number_input("Annual Income", min_value=0, value=50000)
    monthly_income = annual_income / 12

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=15000)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
    debt_to_income_ratio = st.number_input("Debt to Income Ratio (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Debt consolidation", "Medical"])

# These are additional columns your model expects (using default/average values)
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'marital_status': [marital_status],
    'education_level': [education],
    'annual_income': [annual_income],
    'monthly_income': [monthly_income],
    'employment_status': ['Employed'], # Default
    'debt_to_income_ratio': [debt_to_income_ratio],
    'credit_score': [credit_score],
    'loan_amount': [loan_amount],
    'loan_purpose': [loan_purpose],
    'interest_rate': [10.5], # Average placeholder
    'loan_term': [loan_term],
    'installment': [loan_amount / loan_term],
    'grade_subgrade': ['B1'], # Default
    'num_of_open_accounts': [5],
    'total_credit_limit': [annual_income * 1.5],
    'current_balance': [loan_amount * 0.5],
    'delinquency_history': [0],
    'public_records': [0],
    'num_of_delinquencies': [0]
})

if st.button("Predict Loan Status"):
    try:
        prediction = model.predict(input_data)
        
        st.subheader("Result:")
        if prediction[0] == 1:
            st.success("🎉 Congratulations! The loan is likely to be **APPROVED**.")
        else:
            st.error("⚠️ Sorry, the loan is likely to be **REJECTED**.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("AI Data Engineer Project | Loan Approval Simulation")
