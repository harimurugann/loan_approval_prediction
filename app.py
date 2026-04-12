import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

# 1. Page Configuration
st.set_page_config(page_title="Loan Eligibility Platform", layout="wide")

# 2. Model Loading
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HELPER FUNCTIONS ---

def create_pdf(name, result, chance, income, debt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Loan Eligibility Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Applicant Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Approval Chance: {chance}%", ln=True)
    pdf.cell(200, 10, txt=f"Status: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Monthly Disposable Income: ${income}", ln=True)
    pdf.cell(200, 10, txt=f"Monthly Debt: ${debt}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="Note: This is an AI-generated report. Final verification is subject to bank official's approval.")
    return pdf.output(dest='S').encode('latin-1')

def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = pd.DataFrame({
        'Timestamp': [timestamp],
        'Name': [name],
        'Annual_Income': [income],
        'Credit_Score': [credit],
        'Loan_Amount': [amount],
        'Result': [result],
        'Probability': [prob]
    })
    
    if not os.path.exists(log_file):
        log_entry.to_csv(log_file, index=False)
    else:
        log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- THE NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Individual Assessment", 
    "📂 Bulk Processing", 
    "📈 Analytics & Simulator", 
    "🔐 Admin Control Center"
])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tab1:
    st.header("Single Applicant Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "Guest")
        age = st.number_input("Age", 18, 100, 30)
        annual_income = st.number_input("Annual Income ($)", 0, value=45000)
        monthly_income = annual_income / 12
    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 710)
        loan_amount = st.number_input("Loan Amount Requested ($)", 0, value=18000)
        debt_to_income_ratio = st.number_input("DTI Ratio", 0.0, 1.0, 0.15)

    # Feature Engineering (Backend)
    monthly_debt = monthly_income * debt_to_income_ratio
    disposable_income = monthly_income - monthly_debt
    lti = loan_amount / (annual_income if annual_income > 0 else 1)

    # DataFrame creation (Full feature list based on model)
    input_df = pd.DataFrame({
        'age':[age], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
        'annual_income':[annual_income], 'monthly_income':[monthly_income], 'employment_status':['Employed'],
        'debt_to_income_ratio':[debt_to_income_ratio], 'credit_score':[credit_score], 'loan_amount':[loan_amount],
        'loan_purpose':['Business'], 'interest_rate':[10.5], 'loan_term':[36],
        'installment':[loan_amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
        'total_credit_limit':[annual_income*1.5], 'current_balance':[loan_amount*0.5], 'delinquency_history':[0],
        'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[monthly_debt], 
        'disposable_income':[disposable_income], 'loan_to_income_ratio':[lti]
    })

    if st.button("Predict Loan Eligibility"):
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        
        # Save values in session for Analytics Tab
        st.session_state['chance'] = chance
        st.session_state['last_score'] = credit_score
        
        res_text = "APPROVED" if chance >= 50 else "REJECTED"
        
        # Log to Database (Backend)
        log_user_data(name, annual_income, credit_score, loan_amount, res_text, chance)
        st.toast(f"Data for {name} logged successfully", icon="💾")

        if chance >= 70: st.success(f"Approval Probability: {chance}% (Strong Profile)")
        elif chance >= 40: st.warning(f"Approval Probability: {chance}% (Moderate Risk)")
        else: st.error(f"Approval Probability: {chance}% (High Risk Profile)")
        
        pdf_bytes = create_pdf(name, res_text, chance, round(disposable_income,2), round(monthly_debt,2))
        st.download_button("📥 Download Official Report", pdf_bytes, f"{name}_Loan_Report.pdf", "application/pdf")

# --- TAB 2: BULK PROCESSING ---
with tab2:
    st.header("Bulk CSV Processor")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df_bulk = pd.read_csv(uploaded_file)
        st.write("### Preview of Bulk Data:")
        st.dataframe(df_bulk.head())
        # Bulk prediction logic implementation needed here based on model

# --- TAB 3: ANALYTICS & SIMULATOR (FIXED!) ---
with tab3:
    st.header("Decision Engine & Live Simulation")
    
    current_chance = st.session_state.get('chance', 50)
    
    if current_chance >= 70: color, status = "#00CC96", "SAFE / LOW RISK"
    elif current_chance >= 40: color, status = "#FFAA00", "MODERATE RISK"
    else: color, status = "#FF4B4B", "HIGH RISK / REJECTION"

    # 1. Visualization
    st.subheader(f"Current Profile Assessment: :{color}[{status}]")
    imp_df = pd.DataFrame({'Feature': ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Age'], 'Importance %': [45, 25, 15, 10, 5]})
    fig_bar = px.bar(imp_df, x='Importance %', y='Feature', orientation='h', title="Model's key drivers")
    fig_bar.update_traces(marker_color=color)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Reasons Analysis
    st.markdown("### 🔍 Root Cause Analysis")
    reasons = []
    # Tab 1 variables globally accessible aahave irrukum
    if credit_score < 600: reasons.append(f"❌ Low Credit Score ({credit_score}).")
    if debt_to_income_ratio > 0.45: reasons.append(f"❌ High DTI ({debt_to_income_ratio}).")
    
    if reasons:
        st.write("Potential rejection reasons:")
        for r in reasons: st.write(r)
    else:
        st.write("✅ Profile looks financially stable.")

    # 3. What-If Simulator
    st.markdown("---")
    st.subheader("🎯 What-If Approval Simulator")
    s_score = st.slider("Simulate Higher Credit Score", 300, 900, int(credit_score))
    sim_input = input_df.copy()
    sim_input['credit_score'] = [s_score]
    s_prob = model.predict_proba(sim_input)[0][1]
    s_chance = round(s_prob * 100, 2)
    st.metric("Simulated Probability", f"{s_chance}%")

# --- TAB 4: ADMIN SECTION (FIXED!) ---
with tab4:
    st.header("🔐 Admin Data Center")
    
    # 1. Secured Login Gate
    ADMIN_PASSWORD = "admin123" # Secure pannanum na environment variable use pannanum
    password_input = st.text_input("Enter Admin Password", type="password")
    
    if password_input == ADMIN_PASSWORD:
        st.success("Access Granted. Welcome Admin.")
        
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv('user_logs.csv')
            
            # 2. Technical Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Assessments", len(df_logs))
            c2.metric("Approval Rate", f"{round((df_logs['Prediction'] == 'APPROVED').mean() * 100, 2)}%")
            c3.metric("Avg. Credit Score", round(df_logs['Credit_Score'].mean(), 1))

            # 3. Logs Table
            st.subheader("Detailed System Logs")
            st.dataframe(df_logs, use_container_width=True)
            
            # Download Action
            csv_data = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Logs to CSV", csv_data, "system_logs.csv", "text/csv")
            
            if st.button("🗑️ Clear Logs"):
                os.remove('user_logs.csv')
                st.rerun()
        else:
            st.info("System is ready. Awaiting first user input.")
            
    elif password_input == "":
        st.info("Please enter the password to unlock this tab.")
    else:
        st.error("❌ Invalid Password.")

st.markdown("---")
st.caption("Developed by AI Data Engineer | Financial Analytics Platform")
