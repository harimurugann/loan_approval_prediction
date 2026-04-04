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

if st.button("Predict"):
    # 1. Create a dictionary with ALL 21 features in the EXACT order as training
    input_data = {
        'age': age,
        'gender': 1, 'marital_status': 1, 'education_level': 1,
        'annual_income': annual_income,
        'monthly_income': annual_income / 12,
        'employment_status': 1, 
        'debt_to_income_ratio': debt_ratio,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'loan_purpose': 2, 'interest_rate': 12.0, 'loan_term': 36,
        'installment': loan_amount / 36,
        'grade_subgrade': 10, 'num_of_open_accounts': 5,
        'total_credit_limit': annual_income * 1.5,
        'current_balance': loan_amount * 0.5,
        'delinquency_history': 0, 'public_records': 0, 'num_of_delinquencies': 0
    }

    # 2. Convert to DataFrame (This ensures column names and order match the model)
    input_df = pd.DataFrame([input_data])
    
    # 3. Prediction - Use input_df instead of features
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("✅ The model predicts the loan will be PAID BACK.")
    else:
        st.error("⚠️ RISK DETECTED: The model predicts a potential DEFAULT.")
        
