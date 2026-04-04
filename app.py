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
    # Dummy placeholder for other features (since our model expects 21 inputs)
    # You should ideally capture all inputs or use a pre-set default
    features = np.zeros((1, 21)) 
    features[0, 0] = age
    features[0, 4] = annual_income
    features[0, 8] = credit_score
    features[0, 9] = loan_amount
    features[0, 7] = debt_ratio
    
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("The model predicts the loan will be PAID BACK.")
    else:
        st.error("The model predicts the loan will NOT be paid back.")
          
