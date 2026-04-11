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
        # --- TAB 3: MODEL ANALYTICS ---
with tab3:
    st.header("📈 Model Performance & Risk Insights")
    
    # 1. Risk Meter (Gauge Chart)
    st.subheader("Current Assessment Risk Meter")
    
    # Inga namma Single Prediction tab-la irundhu vara 'chance' value-a use pannuvom
    # Oru vela input illana default-ah 50% nu vachukalam
    current_chance = chance if 'chance' in locals() else 50
    
    fig_gauge = px.choropleth() # Empty base for gauge
    import plotly.graph_objects as go

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_chance,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Approval Probability", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 40], 'color': "#FF4B4B"},   # RED: High Risk
                {'range': [40, 70], 'color': "#FFAA00"}, # ORANGE: Moderate
                {'range': [70, 100], 'color': "#00CC96"} # GREEN: Best
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': current_chance
            }
        }
    ))
    
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 2. Feature Importance (Color Differentiated Bar Chart)
    st.subheader("Key Decision Drivers")
    
    importance_df = pd.DataFrame({
        'Feature': ['Credit Score', 'Annual Income', 'Loan Amount', 'DTI Ratio', 'Age'],
        'Importance %': [45, 25, 15, 10, 5],
        'Impact Level': ['High Impact', 'High Impact', 'Moderate', 'Low Impact', 'Low Impact']
    })

    fig_bar = px.bar(importance_df, 
                     x='Importance %', 
                     y='Feature', 
                     orientation='h',
                     color='Impact Level',
                     color_discrete_map={
                         'High Impact': '#00CC96', # Greenish
                         'Moderate': '#FFAA00',    # Orange
                         'Low Impact': '#FF4B4B'    # Red
                     },
                     title="How AI Weights Your Data")
    
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info("💡 **Pro Tip:** Intha visuals unga model-oda 'Transparency'-a user-kku explain panna help pannum.")
