import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline
model = joblib.load('loan_model_pipeline.sav')

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant details to check loan eligibility.")

# Create input fields based on dataset columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    income = st.number_input("Annual Income", min_value=0, value=50000)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=15000)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
    dti = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.1)
    purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Debt consolidation", "Medical"])

# Prepare input data
input_dict = {
    'age': age, 'gender': gender, 'marital_status': marital_status,
    'education_level': education, 'annual_income': income,
    'debt_to_income_ratio': dti, 'credit_score': credit_score,
    'loan_amount': loan_amount, 'loan_purpose': purpose,
    'loan_term': loan_term,
    # Adding placeholders for other columns used in training
    'monthly_income': income/12, 'employment_status': 'Employed',
    'interest_rate': 10.0, 'installment': 500, 'grade_subgrade': 'B1',
    'num_of_open_accounts': 5, 'total_credit_limit': 50000,
    'current_balance': 10000, 'delinquency_history': 0,
    'public_records': 0, 'num_of_delinquencies': 0
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Loan Status"):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("🎉 Congratulations! The loan is likely to be APPROVED.")
    else:
        st.error("⚠️ Sorry, the loan is likely to be REJECTED.")
