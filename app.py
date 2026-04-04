import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model
# Make sure 'loan_model.sav' is in the same folder
model = joblib.load('loan_model.sav')

# App Title
st.title("🚀 Loan Approval Prediction App")
st.write("Enter the customer details below to check loan eligibility:")

# 2. User Input Section
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Annual Income ($)", min_value=1000, value=50000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

with col2:
    loan_amount = st.number_input("Loan Amount ($)", min_value=500, value=15000)
    debt_ratio = st.number_input("Debt to Income Ratio (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# 3. Prediction Logic
if st.button("Predict Loan Status"):
    # Creating the input dictionary with all 21 features
    # Indentation is strictly handled here
    input_data = {
        'age': age,
        'gender': 1, 
        'marital_status': 1, 
        'education_level': 1,
        'annual_income': annual_income,
        'monthly_income': annual_income / 12,
        'employment_status': 1, 
        'debt_to_income_ratio': debt_ratio,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'loan_purpose': 2, 
        'interest_rate': 25.0 if credit_score < 500 else 12.0,
        'loan_term': 60 if credit_score < 500 else 36,
        'installment': loan_amount / 36,
        'grade_subgrade': 25 if credit_score < 500 else 10,
        'num_of_open_accounts': 10 if credit_score < 500 else 5,
        'total_credit_limit': annual_income * 0.7,
        'current_balance': loan_amount * 0.9,
        'delinquency_history': 5 if credit_score < 500 else 0,
        'public_records': 1 if credit_score < 400 else 0,
        'num_of_delinquencies': 5 if credit_score < 500 else 0
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Model Prediction
    prediction = model.predict(df_input)
    
    # Display Result
    st.markdown("---")
    if prediction[0] == 1:
        st.success("✅ **Result: APPROVED** - The model predicts the loan will be PAID BACK.")
    else:
        st.error("⚠️ **Result: REJECTED** - RISK DETECTED! Potential for Default.")
        
