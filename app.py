import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('loan_model.sav')

st.title("Loan Approval Prediction App")
st.write("Enter the following details to predict loan repayment status:")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=1000, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
loan_amount = st.number_input("Loan Amount", min_value=500, value=15000)
debt_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.2)

    # Modified input data to trigger Risk more effectively
    input_data = {
        'age': age,
        'gender': 1, 'marital_status': 1, 'education_level': 1,
        'annual_income': annual_income,
        'monthly_income': annual_income / 12,
        'employment_status': 1, 
        'debt_to_income_ratio': debt_ratio,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'loan_purpose': 2, 
        'interest_rate': 25.0 if credit_score < 500 else 12.0, # High risk if low score
        'loan_term': 60, # Longer term usually means higher risk
        'installment': loan_amount / 60,
        'grade_subgrade': 25 if credit_score < 500 else 10, # Lower grade for low score
        'num_of_open_accounts': 10,
        'total_credit_limit': annual_income * 0.8,
        'current_balance': loan_amount * 0.9,
        'delinquency_history': 5 if credit_score < 500 else 0, # Added history of defaults
        'public_records': 1 if credit_score < 400 else 0,
        'num_of_delinquencies': 5 if credit_score < 500 else 0
    }

    if prediction[0] == 1:
        st.success("✅ The model predicts the loan will be PAID BACK.")
    else:
        st.error("⚠️ RISK DETECTED: The model predicts a potential DEFAULT.")
        
