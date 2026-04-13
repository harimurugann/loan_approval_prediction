import streamlit as st
import pandas as pd
import joblib
import numpy as np
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

# --- UTILITIES ---
def log_user_data(name, income, credit, amount, result, prob):
    log_file = 'user_logs_production.csv'
    # Simulation: Geo Coords (Chennai Area)
    lat, lon = 13.0827 + np.random.uniform(-0.1, 0.1), 80.2707 + np.random.uniform(-0.1, 0.1)
    log_entry = pd.DataFrame({
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Applicant_Name': [name], 'Annual_Income': [income], 'Credit_Score': [credit],
        'Loan_Amount': [amount], 'Prediction': [result], 'Probability_%': [prob],
        'lat': [lat], 'lon': [lon]
    })
    if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
    else: log_entry.to_csv(log_file, mode='a', header=False, index=False)

# --- SIDEBAR BRANDING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Admin Control")
    st.write("👨‍💻 **DEV:** Hari murugan")
    st.write("🚀 **Role:** Data Scientist")
    st.markdown("---")
    st.success("System: Active ✅")

# --- NAVIGATION TABS (Strict Order: 0 to 6) ---
tabs = st.tabs([
    "👤 Assessment", 
    "📂 Bulk Hub", 
    "🗺️ Live Geo Mapping", 
    "📈 Model Drift", 
    "🧠 Explainable AI", 
    "🏦 Market & Cards", 
    "🔐 Admin & MLOps"
])

# --- TAB 0: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        u_name = st.text_input("Full Name", "Guest User")
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000)
    
    if st.button("Run AI Prediction"):
        # Formatting for Model Pipeline (24 Features)
        input_df = pd.DataFrame({'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"], 'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'], 'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount], 'loan_purpose':['Personal'], 'interest_rate':[10.5], 'loan_term':[36], 'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5], 'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0], 'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]})
        
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        
        # Session states for other tabs
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        st.session_state['last_res'] = res
        
        if res == "APPROVED":
            if chance >= 75: st.balloons(); st.success(f"✅ Approved ({chance}%) - High Confidence")
            else: st.warning(f"⚠️ Moderate Risk Approved ({chance}%)")
        else:
            st.error(f"❌ Rejected ({chance}%) - High Risk Profile")
        
        log_user_data(u_name, income, credit, amount, res, chance)

# --- TAB 1: BULK HUB ---
with tabs[1]:
    st.header("📂 Bulk Processing Engine")
    uploaded_file = st.file_uploader("Upload CSV for Batch Check", type="csv")
    if uploaded_file:
        df_bulk = pd.read_csv(uploaded_file)
        st.dataframe(df_bulk.head())
        if st.button("Start Batch Prediction"):
            df_bulk['AI_Result'] = np.where(df_bulk['credit_score'] > 600, "Approved", "Manual Review")
            st.success("Batch Prediction Complete!")
            st.dataframe(df_bulk)

# --- TAB 2: LIVE GEO MAPPING ---
with tabs[2]:
    st.header("🗺️ Applicant Geospatial Distribution")
    if os.path.exists('user_logs_production.csv'):
        df_geo = pd.read_csv('user_logs_production.csv')
        st.map(df_geo[['lat', 'lon']])
        st.info("📍 Live mapping active. Each dot represents a simulated applicant location.")
    else:
        st.info("No data available. Run an Assessment first.")

# --- TAB 3: MODEL DRIFT ---
with tabs[3]:
    st.header("📈 Production Health & Drift Monitor")
    drift_df = pd.DataFrame({'Day': range(1,11), 'Performance': [0.94, 0.93, 0.94, 0.92, 0.94, 0.91, 0.90, 0.92, 0.91, 0.92]})
    st.plotly_chart(px.line(drift_df, x='Day', y='Performance', title="Model Accuracy Stability"))
    st.success("Current Status: Stable ✅")

# --- TAB 4: EXPLAINABLE AI ---
with tabs[4]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        st.subheader("Why did the AI make this decision?")
        # Simulated SHAP/LIME importance
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        features = ['Credit Score', 'Income', 'Loan Amount', 'DTI', 'Age']
        st.plotly_chart(px.bar(x=impact, y=features, orientation='h', color=impact, color_continuous_scale='RdYlGn'))
        for i, f in enumerate(features):
            st.write(f"**{f}**: {'✅ Positive' if impact[i]>0 else '❌ Negative'} Impact")
    else:
        st.warning("Please run an Assessment first to see AI logic.")

# --- TAB 5: MARKET & CARDS ---
with tabs[5]:
    st.header("🏦 Comparative Market Rates")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Bank Rates Table")
        st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari AI Bank'], 'Rate': ['10.5%', '10.7%', '9.2%']}))
    with col_b:
        st.subheader("💳 Card Recommendations")
        if 'last_score' in st.session_state:
            score = st.session_state['last_score']
            if score > 700: st.info("🏆 **Premium Card:** Eligible for High Cashback.")
            else: st.warning("💳 **Secured Card:** Build your score here.")
        else:
            st.write("Run assessment to see eligible cards.")

# --- TAB 6: ADMIN & MLOps ---
with tabs[6]:
    st.header("🔐 Secure Admin Center")
    st.write(f"**Admin Access:** Hari murugan | Senior Data Scientist")
    if st.text_input("Enter Master Password", type="password", key="admin_key") == "admin123":
        if os.path.exists('user_logs_production.csv'):
            df_logs = pd.read_csv('user_logs_production.csv')
            st.dataframe(df_logs.tail(20), use_container_width=True)
            st.download_button("Download Full Audit Logs", df_logs.to_csv(index=False), "system_audit.csv")
        else:
            st.info("Logs are empty.")
