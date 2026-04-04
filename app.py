import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('loan_model.sav')

st.title("Loan Approval Prediction App")
st.write("Enter the following details to predict loan repayment status:")

# Create input fields for user (example fields)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=1000, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
loan_amount = st.number_input("Loan Amount", min_value=500, value=15000)
debt_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.2)

# Since we have many features, for simplicity in this demo, 
# we'll create a dummy array matching the model's expected input shape.
# In a real app, you should add input fields for all 21 features.
if st.button("Predict"):
    # 1. Create a dictionary with ALL 21 features used during training
    # Initializing with average/common values from your dataset
    input_data = {
        'age': age,
        'gender': 1,            # Default: Male
        'marital_status': 1,    # Default: Married
        'education_level': 1,   # Default: Bachelor's
        'annual_income': annual_income,
        'monthly_income': annual_income / 12,
        'employment_status': 1, # Default: Employed
        'debt_to_income_ratio': debt_ratio,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'loan_purpose': 2,      # Default: Debt Consolidation
        'interest_rate': 12.0,  # Average Interest
        'loan_term': 36,        # Standard term
        'installment': loan_amount / 36,
        'grade_subgrade': 10,
        'num_of_open_accounts': 5,
        'total_credit_limit': annual_income * 1.5,
        'current_balance': loan_amount * 0.5,
        'delinquency_history': 0, 
        'public_records': 0,
        'num_of_delinquencies': 0
    }

    # 2. Convert to DataFrame to maintain column order
    input_df = pd.DataFrame([input_data])
    
    # 3. Prediction
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("✅ The model predicts the loan will be PAID BACK.")
    else:
        st.error("⚠️ RISK DETECTED: The model predicts a potential DEFAULT.")
        
