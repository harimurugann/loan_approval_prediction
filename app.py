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
st.set_page_config(page_title="Loan Intelligence Pro", layout="wide")

# 2. Custom CSS for Aesthetic Branding
st.markdown("""
    <style>
    /* Main Background matrum Font maatha */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Buttons-a aesthetic-ah maatha */
    div.stButton > button:first-child {
        background-color: #00CC96;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #00af82;
        border-color: #00af82;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,204,150,0.3);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load the Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- UTILITY FUNCTIONS ---

def create_pdf(name, result, chance, income, debt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Loan Eligibility Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Applicant Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Final Status: {result}", ln=True)
    pdf.cell(200, 10, txt=f"AI Confidence Score: {chance}%", ln=True)
    pdf.cell(200, 10, txt=f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name],
        'Annual_Income': [income],
        'Credit_Score': [credit],
        'Loan_Amount': [amount],
        'Prediction': [result],
        'Probability_%': [prob]
    })
    if not os.path.isfile(log_file):
        log_entry.to_csv(log_file, index=False)
    else:
        log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- SIDEBAR BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Admin Panel")
    st.markdown("---")
    st.write("👨‍💻 **Developer:** Vimal")
    st.write("🚀 **Role:** AI Data Engineer")
    st.markdown("---")
    st.subheader("System Status")
    st.success("Model: Online ✅")
    st.info("Version: 2.1.0 (Guardrails Active)")
    st.markdown("---")
    st.caption("Advanced Loan Eligibility Predictor © 2026")

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Individual Check", 
    "📂 Bulk Processing", 
    "📈 Analytics & Simulator", 
    "🔐 Admin Control"
])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tab1:
    st.header("Single Applicant Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "Guest User")
        age = st.number_input("Age", 18, 100, 30)
        annual_income = st.number_input("Annual Income ($)", 0, value=55000)
        monthly_income = annual_income / 12
    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 720)
        loan_amount = st.number_input("Loan Amount Requested ($)", 0, value=20000)
        debt_to_income = st.number_input("DTI Ratio", 0.0, 1.0, 0.2)

    # Engineering logic
    m_debt = monthly_income * debt_to_income
    disposable = monthly_income - m_debt
    lti = loan_amount / (annual_income if annual_income > 0 else 1)

    input_df = pd.DataFrame({
        'age':[age], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
        'annual_income':[annual_income], 'monthly_income':[monthly_income], 'employment_status':['Employed'],
        'debt_to_income_ratio':[debt_to_income], 'credit_score':[credit_score], 'loan_amount':[loan_amount],
        'loan_purpose':['Business'], 'interest_rate':[10.5], 'loan_term':[36],
        'installment':[loan_amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
        'total_credit_limit':[annual_income*1.5], 'current_balance':[loan_amount*0.5], 'delinquency_history':[0],
        'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[m_debt], 
        'disposable_income':[disposable], 'loan_to_income_ratio':[lti]
    })

    if st.button("Analyze Eligibility"):
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        
        # Guardrail: Credit Score Rejection
        if credit_score < 500:
            res = "REJECTED"
            chance = min(chance, 35.0)
            st.error(f"🚫 High Risk: Credit Score {credit_score} is below threshold.")
        else:
            res = "APPROVED" if chance >= 50 else "REJECTED"

        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit_score
        
        log_user_data(name, annual_income, credit_score, loan_amount, res, chance)
        
        if res == "APPROVED":
            st.success(f"Final Decision: {res} ({chance}% Confidence)")
        else:
            if credit_score >= 500: st.error(f"Final Decision: {res} ({chance}% Confidence)")
        
        pdf_bytes = create_pdf(name, res, chance, round(disposable,2), round(m_debt,2))
        st.download_button("📥 Download Assessment PDF", pdf_bytes, f"{name}_Report.pdf", "application/pdf")

# --- TAB 2: BULK SYSTEM ---
with tab2:
    st.header("Bulk Processing Hub")
    up_file = st.file_uploader("Upload CSV", type="csv")
    if up_file:
        df_bulk = pd.read_csv(up_file)
        st.dataframe(df_bulk.head())

# --- TAB 3: ANALYTICS & SIMULATOR ---
with tab3:
    st.header("Decision Insights & What-If Simulator")
    cur_chance = st.session_state.get('last_chance', 50)
    
    if cur_chance >= 70: color, status = "#00CC96", "LOW RISK"
    elif cur_chance >= 40: color, status = "#FFAA00", "MODERATE"
    else: color, status = "#FF4B4B", "HIGH RISK"

    st.subheader(f"Current Status: :{color}[{status}]")
    imp_df = pd.DataFrame({'Feature': ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Age'], 'Impact %': [45, 25, 15, 10, 5]})
    fig = px.bar(imp_df, x='Impact %', y='Feature', orientation='h')
    fig.update_traces(marker_color=color)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Live Approval Simulator")
    s_score = st.slider("Simulate Credit Score", 300, 900, int(st.session_state.get('last_score', 720)))
    sim_in = input_df.copy()
    sim_in['credit_score'] = [s_score]
    s_prob = model.predict_proba(sim_in)[0][1]
    s_chance = round(s_prob * 100, 2)
    st.metric("Simulated Probability", f"{s_chance}%", f"{round(s_chance - cur_chance, 2)}%")

# --- TAB 4: ADMIN (SECURED) ---
with tab4:
    st.header("🔐 Secure Data Logs")
    pwd = st.text_input("Enter Admin Password", type="password")
    
    if pwd == "admin123":
        st.success("Authorized Access.")
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv('user_logs.csv')
            
            p_col = 'Prediction' if 'Prediction' in df_logs.columns else 'Result'
            
            c1, c2 = st.columns(2)
            c1.metric("Total Usage", len(df_logs))
            c2.metric("Approval Rate", f"{round((df_logs[p_col] == 'APPROVED').mean() * 100, 2)}%")
            
            st.dataframe(df_logs, use_container_width=True)
            if st.button("🗑️ Clear Logs"):
                os.remove('user_logs.csv')
                st.rerun()
    elif pwd != "":
        st.error("Invalid credentials.")
