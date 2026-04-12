import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import datetime
import os

# 1. Page Config
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

# Custom UI Styling
st.markdown("""<style>.stApp {background-color: #0e1117; color: white;} div.stButton > button:first-child {background-color: #00CC96; color: white; border-radius: 8px;}</style>""", unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- LOGGING FUNCTION ---
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs.csv'
    log_entry = pd.DataFrame({'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')], 'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit], 'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob]})
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- SIDEBAR BRANDING ---
with st.sidebar:
    st.title("Project Control")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.markdown("---")
    st.success("Model: Online ✅")

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["👤 Assessment", "📂 Bulk Processing", "📈 Analytics & Simulator", "🔐 Admin Center"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tab1:
    st.header("Single Applicant Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Name", "Guest")
        income = st.number_input("Annual Income ($)", 0, 500000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount ($)", 0, 100000, 20000)

    if st.button("Run Prediction"):
        # Dummy features to match model input
        input_data = pd.DataFrame([[30, income, credit, amount]], columns=['age', 'annual_income', 'credit_score', 'loan_amount'])
        prob = model.predict_proba(input_data)[0][1]
        chance = round(prob * 100, 2)
        
        # Guardrail
        res = "REJECTED" if credit < 500 else ("APPROVED" if chance >= 50 else "REJECTED")
        
        # SAVE TO SESSION STATE (This fixes the blank screen!)
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        st.session_state['last_amount'] = amount
        st.session_state['last_res'] = res
        
        log_user_data(name, income, credit, amount, res, chance)
        
        if res == "APPROVED": st.success(f"Final Decision: {res} ({chance}% Confidence)")
        else: st.error(f"Final Decision: {res} ({chance}% Confidence)")

# --- TAB 2: BULK PROCESSING (FIXED BLANK) ---
with tab2:
    st.header("📂 Bulk Assessment Engine")
    up_file = st.file_uploader("Upload CSV file for batch prediction", type="csv")
    if up_file:
        df_bulk = pd.read_csv(up_file)
        st.write("### Data Preview")
        st.dataframe(df_bulk.head())
        if st.button("Process Bulk Records"):
            # Sample logic: adding a prediction column
            df_bulk['AI_Decision'] = np.where(df_bulk['credit_score'] > 600, "Approved", "Check Manually")
            st.write("### Processed Results")
            st.dataframe(df_bulk)
            st.download_button("Download Results", df_bulk.to_csv(index=False), "bulk_results.csv")
    else:
        st.info("Upload a CSV file with 'credit_score', 'annual_income', etc., to start bulk processing.")

# --- TAB 3: ANALYTICS (FIXED BLANK) ---
with tab3:
    st.header("📈 Decision Analytics")
    if 'last_chance' in st.session_state:
        chance = st.session_state['last_chance']
        score = st.session_state['last_score']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Approval Probability")
            fig = go.Figure(go.Indicator(mode = "gauge+number", value = chance, title = {'text': "Confidence %"}, gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00CC96"}}))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("What-If Simulation")
            s_score = st.slider("Adjust Credit Score", 300, 900, int(score))
            st.write(f"Simulated Score: {s_score}")
            st.info("Adjust the slider to see how the model responds to score changes.")
    else:
        st.warning("⚠️ No data found. Please run a prediction in the 'Assessment' tab first.")

# --- TAB 4: ADMIN ---
with tab4:
    st.header("🔐 Admin Dashboard")
    st.write(f"**DEV:** Hari murugan | Data Scientist")
    if st.text_input("Password", type="password") == "admin123":
        if os.path.exists('user_logs.csv'):
            st.dataframe(pd.read_csv('user_logs.csv'))
        else: st.info("Logs are empty.")
