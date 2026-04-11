import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px

# Page Setup
st.set_page_config(page_title="Loan Analytics Pro", layout="wide")

# 1. Model Loading
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- PDF Generation Function ---
def create_pdf(name, result, chance, income, debt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Loan Approval Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Applicant Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Status: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Approval Probability: {chance}%", ln=True)
    pdf.cell(200, 10, txt=f"Monthly Disposable Income: ${income}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- CREATING THE TABS ---
# Indha line thaan unga app-a 3-ah pirikkum
tab1, tab2, tab3 = st.tabs(["👤 Single Prediction", "📂 Bulk Upload (CSV)", "📈 Model Analytics"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.header("Individual Loan Eligibility Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Applicant Name", "Guest User")
        age = st.number_input("Age", 18, 100, 30)
        annual_income = st.number_input("Annual Income ($)", 0, value=50000)
        monthly_income = annual_income / 12
    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        loan_amount = st.number_input("Loan Amount Requested ($)", 0, value=15000)
        debt_to_income_ratio = st.number_input("DTI Ratio (0.0 to 1.0)", 0.0, 1.0, 0.1)

    # Feature Engineering
    monthly_debt = monthly_income * debt_to_income_ratio
    disposable_income = monthly_income - monthly_debt
    lti = loan_amount / (annual_income if annual_income > 0 else 1)

    # DataFrame creation (Ensure all 24 columns match your model)
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

    if st.button("Predict & Generate PDF"):
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res_text = "APPROVED" if chance >= 50 else "REJECTED"
        
        if chance >= 70: st.success(f"Approval Probability: {chance}% - Safe Profile")
        elif chance >= 40: st.warning(f"Approval Probability: {chance}% - Moderate Risk")
        else: st.error(f"Approval Probability: {chance}% - High Risk")
        
        pdf_data = create_pdf(name, res_text, chance, round(disposable_income,2), round(monthly_debt,2))
        st.download_button("📥 Download PDF Report", pdf_data, f"{name}_Loan_Report.pdf", "application/pdf")

# --- TAB 2: BULK PREDICTION ---
with tab2:
    st.header("Bulk Processing (CSV)")
    st.write("Upload a CSV file with multiple applicant details.")
    uploaded_file = st.file_uploader("Choose CSV", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        # Apply Feature Engineering to bulk data
        data['monthly_income'] = data['annual_income'] / 12
        data['monthly_debt'] = data['monthly_income'] * data['debt_to_income_ratio']
        data['disposable_income'] = data['monthly_income'] - data['monthly_debt']
        data['loan_to_income_ratio'] = data['loan_amount'] / data['annual_income']
        
        # Batch Prediction
        preds = model.predict(data)
        data['Prediction'] = ["Approved" if p == 1 else "Rejected" for p in preds]
        
        st.write("### Preview of Processed Data:")
        st.dataframe(data.head())
        
        csv_res = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Result CSV", csv_res, "bulk_results.csv", "text/csv")
      # --- TAB 3: MODEL ANALYTICS (REASON ANALYSIS) ---
with tab3:
    st.header("📈 Live Risk Analytics & Reasonings")

    current_chance = chance if 'chance' in locals() else 50

    # --- WHAT-IF SIMULATOR SECTION ---
st.markdown("---")
st.header("🎯 What-If Approval Simulator")
st.write("Current status 'Rejected' or 'Moderate'-ah irundha, kila irukura sliders-a adjust panni status-a approve panna try pannunga.")

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    sim_credit = st.slider("Simulate Credit Score Increase", min_value=300, max_value=900, value=int(credit_score))
    sim_income = st.slider("Simulate Annual Income Increase ($)", min_value=int(annual_income), max_value=int(annual_income + 50000), step=1000)

with col_sim2:
    sim_loan = st.slider("Simulate Lower Loan Amount ($)", min_value=1000, max_value=int(loan_amount), step=500, value=int(loan_amount))

# Calculate updated metrics for the simulator
sim_monthly_income = sim_income / 12
sim_monthly_debt = sim_monthly_income * debt_to_income_ratio # Assuming DTI stays same
sim_disposable = sim_monthly_income - sim_monthly_debt
sim_lti = sim_loan / sim_income

# Create simulation dataframe
sim_df = input_df.copy()
sim_df['credit_score'] = [sim_credit]
sim_df['annual_income'] = [sim_income]
sim_df['loan_amount'] = [sim_loan]
sim_df['monthly_income'] = [sim_monthly_income]
sim_df['disposable_income'] = [sim_disposable]
sim_df['loan_to_income_ratio'] = [sim_lti]

# Predict for simulation
sim_prob = model.predict_proba(sim_df)[0][1]
sim_chance = round(sim_prob * 100, 2)

# Display Simulation Result
st.subheader(f"Simulated Approval Chance: {sim_chance}%")

if sim_chance >= 70:
    st.balloons()
    st.success(f"✅ Success! With a Credit Score of {sim_credit} and Income of ${sim_income}, your loan would likely be **APPROVED**.")
elif sim_chance >= 40:
    st.warning("⚠️ Still in Moderate Risk. Try increasing your Credit Score or decreasing the Loan Amount further.")
else:
    st.error("❌ Still in High Risk. Significant financial improvements are needed.")

    # Status & Colour Logic
    if current_chance >= 70:
        status_colour, status_text = "#00CC96", "SAFE / APPROVED"
    elif current_chance >= 40:
        status_colour, status_text = "#FFAA00", "MODERATE RISK"
    else:
        status_colour, status_text = "#FF4B4B", "HIGH RISK / REJECTION"

    st.subheader(f"Current Status: :{status_colour}[{status_text}]")

    # 1. Dynamic Bar Chart
    importance_df = pd.DataFrame({
        'Feature': ['Credit Score', 'Annual Income', 'Loan Amount', 'DTI Ratio', 'Age'],
        'Importance %': [45, 25, 15, 10, 5]
    })
    fig_bar = px.bar(importance_df, x='Importance %', y='Feature', orientation='h')
    fig_bar.update_traces(marker_color=status_colour) 
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2. REASONING SECTION (New)
    st.markdown("### 🔍 Why this Result?")
    
    reasons = []
    
    # Logic to identify specific red flags
    if credit_score < 600:
        reasons.append(f"❌ **Low Credit Score ({credit_score}):** Most banks require at least 700 for automatic approval.")
    if debt_to_income_ratio > 0.45:
        reasons.append(f"❌ **High Debt-to-Income ({debt_to_income_ratio}):** More than 45% of your income goes to debt, making new loans risky.")
    if loan_amount > (annual_income * 2):
        reasons.append(f"❌ **High Loan-to-Income:** Requested loan is more than twice your annual income.")
    if age < 21:
        reasons.append(f"⚠️ **Young Age ({age}):** Limited credit history might be affecting the confidence score.")

    # Displaying the Reasons
    if status_text == "SAFE / APPROVED":
        st.write("✅ All your financial metrics are within the 'Low Risk' threshold.")
    else:
        for r in reasons:
            st.write(r)
            
    if not reasons and status_text != "SAFE / APPROVED":
        st.write("📝 The model identifies a combination of factors leading to moderate risk. Consider reducing the loan amount for better odds.")

    st.markdown("---")
    st.info("💡 **AI Tip:** Improving the factors marked with ❌ will significantly move the chart towards Green.")
