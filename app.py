import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

# 1. Page Configuration & UI Styling
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;
        padding: 0.6rem 2rem; border: none; transition: 0.3s ease;
    }
    div.stButton > button:first-child:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(0,204,150,0.4); }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00CC96; }
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

# --- SIDEBAR: AI CHATBOT & BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    chat_input = st.text_input("Ask Bot...", placeholder="e.g. How to fix low credit?")
    if chat_input:
        q = chat_input.lower()
        if "credit" in q: st.info("💡 **Bot:** Pay bills on time and keep card usage below 30%.")
        else: st.write("Try asking about 'Credit' or 'Market Rates'.")
    st.markdown("---")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Processing", "📈 Explainable AI", "🏦 Market Rates", "🔐 Admin Center"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
        loan_cat = st.selectbox("Loan Category", ["Personal Loan", "Home Loan", "Car Loan", "Business Loan", "Education Loan"])
        
    with col2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        # FIX: Increased limit to 10 Crore
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000) 
        dti = st.slider("Current DTI Ratio", 0.0, 1.0, 0.25)
    
    if st.button("Run AI Prediction"):
        # Formatting data for model (24 columns fix)
        input_df = pd.DataFrame({
            'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"],
            'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'],
            'debt_to_income_ratio':[dti], 'credit_score':[credit], 'loan_amount':[amount],
            'loan_purpose':[loan_cat], 'interest_rate':[10.5], 'loan_term':[36],
            'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5],
            'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0],
            'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*dti], 
            'disposable_income':[income/12 - (income/12*dti)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]
        })

        # 1. Prediction logic
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        
        # Guardrail logic
        if credit < 500:
            res = "REJECTED"
            chance = min(chance, 30.0)
            st.error(f"🚫 High Risk rejection message.")
        else:
            # FIX: Approval Threshold logic. 54% is > 50%, so it's Approved.
            res = "APPROVED" if chance >= 50 else "REJECTED"
        
        # Save results for Analytics
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        st.session_state['last_res'] = res
        st.session_state['last_cat'] = loan_cat
        
        log_user_data(name, income, credit, amount, res, chance)

        # FIX: Custom Messages based on Result
        if res == "APPROVED":
            if chance >= 75:
                st.balloons()
                st.success(f"🎉 **Congratulations {name}!** Your {loan_cat} is **APPROVED** with high confidence ({chance}%).")
            else:
                st.warning(f"⚠️ **Approved (Moderate Risk):** Your {loan_cat} is approved ({chance}%), but consider improving your credit score for better rates.")
        else:
            if credit < 500:
                st.error(f"❌ **Rejected:** Sorry, your credit score ({credit}) is too low for approval.")
            else:
                st.error(f"❌ **Rejected:** High application risk detected ({chance}% confidence). Check 'Explainable AI' for reasons.")

# --- TAB 3: EXPLAINABLE AI (Description Fixed) ---
with tabs[2]:
    st.header("🧠 Explainable AI (XAI)")
    if 'last_chance' in st.session_state:
        st.subheader(f"AI Decision Logic for {st.session_state['last_cat']}")
        
        # SHAP-like importance logic (Dynamic based on score)
        c_score = st.session_state['last_score']
        impact = [45 if c_score > 600 else -50, 20, -15, 10, 10]
        features = ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Employment']
        
        fig_xai = px.bar(x=impact, y=features, orientation='h', 
                         color=impact, color_continuous_scale='RdYlGn',
                         labels={'x': 'Impact on Approval', 'y': 'Feature'})
        st.plotly_chart(fig_xai, use_container_width=True)
        
        # FIX: Description based on range
        st.markdown("### 🔍 Detailed Feature Description")
        for i, feat in enumerate(features):
            imp = impact[i]
            if imp > 30: 
                st.write(f"✅ **{feat}** ({imp} range): Major positive driver for approval.")
            elif imp > 0:
                st.write(f"🟢 **{feat}** ({imp} range): Minor positive factor.")
            elif imp > -20:
                st.write(f"🟠 **{feat}** ({imp} range): Minor negative impact.")
            else:
                st.write(f"❌ **{feat}** ({imp} range): Critical rejection driver. Must improve.")
    else:
        st.warning("⚠️ No data available. Run Assessment first.")

# --- TAB 4: MARKET RATES (Blank Fix) ---
with tabs[3]:
    st.header("🏦 Real-time Market Rates")
    st.write("Compare eligible interest rates across top banks.")
    
    # Mock data integration
    market_rates = pd.DataFrame({
        'Bank Name': ['SBI', 'HDFC', 'ICICI', 'Axis', 'Hari Bank (AI)'],
        'Personal Loan Rate (%)': [10.5, 10.7, 10.8, 11.0, 9.5],
        'Home Loan Rate (%)': [8.4, 8.6, 8.5, 8.8, 7.9],
        'Processing Fee': ['0.5%', '1%', '0.8%', '1%', '0%']
    })
    
    st.table(market_rates)
    
    # Visualization
    fig_market = px.bar(market_rates, x='Bank Name', y='Personal Loan Rate (%)', 
                        color='Bank Name', title="Personal Loan Rates Comparison")
    st.plotly_chart(fig_market, use_container_width=True)

with tabs[4]:
    st.header("🔐 Admin Center")
    st.write("DEV: Hari murugan | Data Scientist")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            st.dataframe(pd.read_csv('user_logs.csv'))
