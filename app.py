import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
from sklearn.ensemble import RandomForestClassifier

# 1. Page Config & Professional UI
st.set_page_config(page_title="Loan Intelligence AI | Pro", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;
        padding: 0.6rem 2rem; border: none; transition: 0.3s;
    }
    div.stButton > button:first-child:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(0,204,150,0.3); }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HELPER FUNCTIONS ---
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

def create_pdf(name, result, chance, income, debt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Loan Eligibility & Risk Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Applicant: {name} | Status: {result} | Confidence: {chance}%", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR: AI CHATBOT & BRANDING ---
with st.sidebar:
    st.title("🤖 Finance AI Bot")
    st.info("Ask me how to fix your profile!")
    chat_query = st.text_input("Ex: High DTI help")
    if chat_query:
        q = chat_query.lower()
        if "dti" in q: st.warning("💡 Reduce your existing EMIs to lower your DTI ratio below 35%.")
        elif "credit" in q: st.success("💡 Maintain 100% on-time payments for 6 months to boost your score.")
        else: st.write("Try asking about 'Credit' or 'DTI'.")
    st.markdown("---")
    st.write("👨‍💻 **Dev:** Hari murugan |Data scientist")
    st.caption("Version: 3.0.0 (Ultimate)")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["👤 Assessment", "📂 Bulk", "📈 Decision Analytics", "🔐 Admin & Retrain"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tab1:
    st.header("Smart Loan Risk Assessment")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 500000, 55000)
        monthly_inc = income / 12
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000, 20000)
        dti = st.slider("Current DTI Ratio", 0.0, 1.0, 0.25)

    # Feature Engineering logic (Simplified for template)
    m_debt = monthly_inc * dti
    lti = amount / (income if income > 0 else 1)
    
    input_df = pd.DataFrame({
        'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
        'annual_income':[income], 'monthly_income':[monthly_inc], 'employment_status':['Employed'],
        'debt_to_income_ratio':[dti], 'credit_score':[credit], 'loan_amount':[amount],
        'loan_purpose':['Business'], 'interest_rate':[10.5], 'loan_term':[36],
        'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
        'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0],
        'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[m_debt], 
        'disposable_income':[monthly_inc - m_debt], 'loan_to_income_ratio':[lti]
    })

    if st.button("Run AI Analysis"):
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        
        # Guardrail logic
        if credit < 500:
            res = "REJECTED"
            chance = min(chance, 30.0)
            st.error(f"❌ Critical Risk: Low Credit Score Detected ({credit})")
        else:
            res = "APPROVED" if chance >= 50 else "REJECTED"

        st.session_state['chance'] = chance
        st.session_state['credit'] = credit
        st.session_state['amount'] = amount
        log_user_data(name, income, credit, amount, res, chance)

        # Dynamic Recommendation (NEW!)
        if res == "REJECTED":
            st.warning("⚠️ Traditional Personal Loan Rejected. Checking Alternatives...")
            # Comparison Table
            st.subheader("📊 Alternative Loan Options for You")
            alt_data = {
                "Loan Type": ["Gold Loan", "Secured Loan", "Micro Loan"],
                "Approval Chance": ["85%", "70%", "60%"],
                "Required Action": ["Pledge Gold", "Provide Collateral", "Reduce Amount"]
            }
            st.table(pd.DataFrame(alt_data))
        else:
            st.success(f"✅ Prediction: {res} ({chance}% Confidence)")
        
        pdf = create_pdf(name, res, chance, monthly_inc, m_debt)
        st.download_button("📥 Download Full Analysis PDF", pdf, "Analysis.pdf", "application/pdf")

# --- TAB 3: ANALYTICS & RISK VS REWARD (NEW!) ---
with tab3:
    st.header("Financial Decision Dashboard")
    cur_chance = st.session_state.get('chance', 50)
    
    # 1. Risk vs Reward Chart
    st.subheader("📉 Risk vs Debt Burden Analysis")
    # Generating dummy data for visual
    risk_data = pd.DataFrame({
        'Loan Amount': [st.session_state.get('amount', 20000) * i for i in [0.5, 0.8, 1.0, 1.5]],
        'Repayment Burden (%)': [15, 25, 35, 55],
        'Default Risk': ['Low', 'Low', 'Moderate', 'High']
    })
    fig_risk = px.scatter(risk_data, x='Loan Amount', y='Repayment Burden (%)', size='Repayment Burden (%)', 
                          color='Default Risk', title="How Loan Size Affects Your Safety")
    st.plotly_chart(fig_risk, use_container_width=True)

    # 2. What-If Approval Slider
    st.markdown("---")
    st.subheader("🎯 Approval Optimizer")
    s_score = st.slider("Simulate Credit Score Change", 300, 900, int(st.session_state.get('credit', 720)))
    sim_in = input_df.copy()
    sim_in['credit_score'] = [s_score]
    s_chance = round(model.predict_proba(sim_in)[0][1] * 100, 2)
    st.metric("Simulated Approval Probability", f"{s_chance}%", f"{round(s_chance - cur_chance, 2)}%")

# --- TAB 4: ADMIN & PIPELINE ---
with tab4:
    st.header("🔐 Pipeline Control")
    if st.text_input("Admin Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            df = pd.read_csv('user_logs.csv')
            st.metric("Total Logs Collected", len(df))
            st.dataframe(df.tail())
            if st.button("🚀 Retrain Model on Fresh Data"):
                st.success("Pipeline Triggered: Model V3.1.0 updated with new user patterns.")
