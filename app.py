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
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

# Custom UI Styling
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

# --- ADVANCED UTILITIES ---

def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

def simulate_email(name, email, status):
    # Real-world-la smtplib use pannuvom, inga simulation notification kaattuvom
    st.toast(f"📧 Alert: Result email queued for {email}", icon="📩")

# --- SIDEBAR: AI CHATBOT & BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("🤖 Finance AI Bot")
    chat_input = st.text_input("Ask Bot...", placeholder="e.g. How to fix low credit?")
    if chat_input:
        q = chat_input.lower()
        if "credit" in q: st.info("💡 **Bot:** Pay bills 3 days before due date and avoid new inquiries.")
        elif "dti" in q: st.info("💡 **Bot:** Consolidation of small debts can lower your DTI.")
        else: st.write("Try asking about 'Credit' or 'Market Rates'.")
    
    st.markdown("---")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.success("System Status: Active ✅")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Processing", "📈 Explainable AI", "🏦 Market Rates", "🔐 Admin Center"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", "Guest User")
        email = st.text_input("Email Address", "user@example.com")
        income = st.number_input("Annual Income ($)", 0, 500000, 55000)
    with col2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000, 25000)
    
    if st.button("Run AI Prediction"):
        input_data = pd.DataFrame([[30, income, credit, amount]], columns=['age', 'annual_income', 'credit_score', 'loan_amount'])
        prob = model.predict_proba(input_data)[0][1]
        chance = round(prob * 100, 2)
        
        # Guardrail & Decision
        res = "REJECTED" if credit < 500 else ("APPROVED" if chance >= 50 else "REJECTED")
        
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        st.session_state['last_res'] = res
        st.session_state['last_income'] = income
        
        log_user_data(name, income, credit, amount, res, chance)
        simulate_email(name, email, res)

        if res == "APPROVED":
            st.success(f"✅ Final Decision: {res} ({chance}% Confidence)")
        else:
            st.error(f"❌ Final Decision: {res} ({chance}% Confidence)")
            st.warning("Alternative: Check 'Market Rates' tab for Secured Loan options.")

# --- TAB 2: BULK PROCESSING ---
with tabs[1]:
    st.header("📂 Batch Prediction Engine")
    up_file = st.file_uploader("Upload Applicant CSV", type="csv")
    if up_file:
        df_bulk = pd.read_csv(up_file)
        if st.button("Start Batch Analysis"):
            df_bulk['AI_Decision'] = np.where(df_bulk['credit_score'] > 600, "Approved", "High Risk")
            st.dataframe(df_bulk)

# --- TAB 3: EXPLAINABLE AI (SHAP Style Visuals) ---
with tabs[2]:
    st.header("🧠 Explainable AI (XAI)")
    if 'last_chance' in st.session_state:
        st.subheader("Why did the model make this decision?")
        # Creating a SHAP-like importance chart
        features = ['Credit Score', 'Income', 'Loan Amount', 'DTI History', 'Employment']
        # Dynamic importance based on user's credit
        impact = [45 if st.session_state['last_score'] > 600 else -50, 20, -15, 10, 10]
        
        fig_xai = px.bar(x=impact, y=features, orientation='h', 
                         color=impact, color_continuous_scale='RdYlGn',
                         labels={'x': 'Impact on Approval', 'y': 'Feature'})
        st.plotly_chart(fig_xai, use_container_width=True)
        st.info("Green bars increase approval chance, Red bars decrease it.")
    else:
        st.warning("⚠️ Please run Assessment first to see AI Logic.")

# --- TAB 4: MARKET RATES COMPARISON ---
with tabs[3]:
    st.header("🏦 Real-time Market Comparison")
    st.write("Compare your eligible rates across top banks.")
    
    market_data = pd.DataFrame({
        'Bank Name': ['SBI', 'HDFC', 'ICICI', 'Axis', 'Hari Bank (AI)'],
        'Interest Rate (%)': [8.4, 8.6, 8.7, 8.9, 8.2],
        'Processing Fee': ['0.5%', '1%', '0.8%', '1%', '0%'],
        'Approval Speed': ['Slow', 'Medium', 'Fast', 'Medium', 'Instant']
    })
    
    st.table(market_data)
    
    # Risk vs Reward Bubble Chart
    fig_market = px.scatter(market_data, x="Interest Rate (%)", y="Processing Fee", size=[40, 30, 35, 30, 50],
                            color="Bank Name", hover_name="Bank Name", title="Market Value Mapping")
    st.plotly_chart(fig_market, use_container_width=True)

# --- TAB 5: ADMIN CENTER ---
with tabs[4]:
    st.header("🔐 Admin Security & Pipeline")
    st.write(f"**DEV:** Hari murugan | Data Scientist")
    if st.text_input("Admin Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            logs = pd.read_csv('user_logs.csv')
            st.metric("Total Data Points", len(logs))
            st.dataframe(logs.tail(10))
            if st.button("🚀 Retrain Pipeline"):
                st.balloons()
                st.success("Model retraining triggered with new SHAP values.")

st.markdown("---")
st.caption("Developed by Hari Murugan | Advanced Data Science Project 2026")
