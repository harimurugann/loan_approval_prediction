import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Loan Approval Predictor Pro", layout="centered")

# 1. Load the updated pipeline
@st.cache_resource
def load_model():
    # Make sure this file name matches exactly with your GitHub file
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🏦 Advanced Loan Approval Prediction")
st.write("Enter applicant details to get a real-time risk assessment.")

# 2. Input Fields - Organized in 2 columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    monthly_income = annual_income / 12

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=15000)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
    debt_to_income_ratio = st.number_input("Debt to Income Ratio (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.1)
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Medical", "Debt consolidation"])

# 3. --- FEATURE ENGINEERING (Must match your Notebook logic) ---
# Calculating the 3 New Features
monthly_debt = monthly_income * debt_to_income_ratio
disposable_income = monthly_income - monthly_debt
# Avoid division by zero
loan_to_income_ratio = loan_amount / (annual_income if annual_income > 0 else 1)

# 4. Create DataFrame with ALL features (24 columns total)
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'marital_status': [marital_status],
    'education_level': [education],
    'annual_income': [annual_income],
    'monthly_income': [monthly_income],
    'employment_status': ['Employed'], # Default placeholder
    'debt_to_income_ratio': [debt_to_income_ratio],
    'credit_score': [credit_score],
    'loan_amount': [loan_amount],
    'loan_purpose': [loan_purpose],
    'interest_rate': [10.5], # Placeholder
    'loan_term': [loan_term],
    'installment': [loan_amount / loan_term],
    'grade_subgrade': ['B1'], # Placeholder
    'num_of_open_accounts': [5],
    'total_credit_limit': [annual_income * 1.5],
    'current_balance': [loan_amount * 0.5],
    'delinquency_history': [0],
    'public_records': [0],
    'num_of_delinquencies': [0],
    # Adding the 3 New Features:
    'monthly_debt': [monthly_debt],
    'disposable_income': [disposable_income],
    'loan_to_income_ratio': [loan_to_income_ratio]
})

# 5. Prediction and Visualization
if st.button("Analyze Loan Risk"):
    try:
        # Get Probability
        prob = model.predict_proba(input_data)[0][1]
        approval_chance = round(prob * 100, 2)
        
        st.markdown("---")
        st.subheader("📊 Results & Risk Analysis")
        
        # Risk Meter Colors (Green/Yellow/Red)
        if approval_chance >= 70:
            st.success(f"🎉 **High Approval Chance: {approval_chance}%**")
            st.progress(prob)
            st.balloons()
        elif approval_chance >= 40:
            st.warning(f"⚠️ **Moderate Risk: {approval_chance}% Chance**")
            st.progress(prob)
        else:
            st.error(f"❌ **High Risk of Rejection: {approval_chance}% Chance**")
            st.progress(prob)

        # Insights based on New Features
        st.info(f"💡 **Quick Insight:** After current debts, your monthly disposable income is approximately **${round(float(disposable_income), 2)}**.")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}. Please ensure the .sav file is updated with 24 features.")

st.markdown("---")
st.caption("AI Data Engineer Project | Advanced Loan Risk Analytics")
