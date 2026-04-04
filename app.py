import streamlit as st
import pandas as pd
import joblib

# 1. Load the full pipeline
# Idhula preprocessing + model renduமே irukku
model = joblib.load('loan_approval_pipeline.sav')

st.title("Loan Approval Prediction App")

# 2. Get User Input
# Text inputs (Example: 'Gender', 'Loan Purpose') neenga direct-ah vangalam
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Married", "Single"])
income = st.number_input("Monthly Income", min_value=0)
# ... unga matha input fields ...

# 3. Predict Button
if st.button("Predict"):
    # Create a DataFrame for the input
    # Pipeline automatically handles encoding and scaling!
    input_data = pd.DataFrame([[gender, married, income]], 
                              columns=['gender', 'marital_status', 'monthly_income'])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected!")
        
