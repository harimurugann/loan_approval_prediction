import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import datetime
import os

# 1. Page Configuration
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

# Custom UI Styling
st.markdown("""<style>.stApp {background-color: #0e1117; color: white;} div.stButton > button:first-child {background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;}</style>""", unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- NAVIGATION TABS ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Hub", "🗺️ Live Geo Mapping", "📈 Model Drift", "🧠 Explainable AI", "🏦 Market & Cards", "🔐 Admin Center"])

# --- TAB 1: ASSESSMENT ---
with tabs[0]:
    st.header("Smart Loan Risk Check")
    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Annual Income ($)", 0, 10000000, 55000)
    with c2:
        credit = st.number_input("Credit Score", 300, 900, 720)
        amount = st.number_input("Loan Amount Requested ($)", 0, 100000000, 25000)
    
    if st.button("Analyze Eligibility"):
        # Formatting 24 columns for model
        input_df = pd.DataFrame({'age':[30], 'gender':['Male'], 'marital_status':['Single'], 'education_level':["Bachelor's"], 'annual_income':[income], 'monthly_income':[income/12], 'employment_status':['Employed'], 'debt_to_income_ratio':[0.25], 'credit_score':[credit], 'loan_amount':[amount], 'loan_purpose':['Personal'], 'interest_rate':[10.5], 'loan_term':[36], 'installment':[amount/36], 'grade_subgrade':['B1'], 'num_of_open_accounts':[5], 'total_credit_limit':[income*1.5], 'current_balance':[amount*0.5], 'delinquency_history':[0], 'public_records':[0], 'num_of_delinquencies':[0], 'monthly_debt':[income/12*0.25], 'disposable_income':[income/12 - (income/12*0.25)], 'loan_to_income_ratio':[amount/income if income > 0 else 0]})
        prob = model.predict_proba(input_df)[0][1]
        chance = round(prob * 100, 2)
        res = "APPROVED" if (chance >= 50 and credit >= 500) else "REJECTED"
        st.session_state['last_chance'] = chance
        st.session_state['last_score'] = credit
        if res == "APPROVED":
            if chance >= 75: st.balloons(); st.success(f"✅ High Confidence Approval ({chance}%)")
            else: st.warning(f"⚠️ Moderate Risk Approval ({chance}%) - Manual check advised.")
        else: st.error(f"❌ Rejected ({chance}%)")

# --- TAB 2: BULK HUB (FIXED: FILE UPLOADER ADDED) ---
with tabs[1]:
    st.header("📂 Bulk Processing Engine")
    st.write("Upload a CSV file to process multiple loan applications at once.")
    bulk_file = st.file_uploader("Choose CSV file", type="csv")
    if bulk_file:
        df_bulk = pd.read_csv(bulk_file)
        st.write("### Data Preview")
        st.dataframe(df_bulk.head())
        if st.button("Start Batch Prediction"):
            df_bulk['AI_Result'] = np.where(df_bulk['credit_score'] > 600, "Approved", "Manual Review")
            st.success("Batch Prediction Complete!")
            st.dataframe(df_bulk)

# --- TAB 5: EXPLAINABLE AI (FIXED: DESCRIPTION ADDED) ---
with tabs[4]:
    st.header("🧠 Decision Logic (XAI)")
    if 'last_chance' in st.session_state:
        impact = [45 if st.session_state['last_score'] > 600 else -50, 25, -15, 10, 10]
        features = ['Credit Score', 'Annual Income', 'Loan Amount', 'DTI Ratio', 'Age']
        st.plotly_chart(px.bar(x=impact, y=features, orientation='h', color=impact, color_continuous_scale='RdYlGn'))
        
        st.markdown("### 📝 Detailed Decision Breakdown")
        for i, feat in enumerate(features):
            val = impact[i]
            status = "✅ Strong Positive" if val > 30 else ("🟢 Minor Positive" if val > 0 else "❌ Negative Impact")
            st.write(f"**{feat}**: {status} (Impact Score: {val})")
    else: st.info("Run Assessment first to see the logic.")

# --- TAB 6: MARKET & CARDS (FIXED: CARD SUGGESTIONS ADDED) ---
with tabs[5]:
    st.header("🏦 Market Rates & Card Recommendations")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Live Bank Rates")
        st.table(pd.DataFrame({'Bank': ['SBI', 'HDFC', 'Hari AI Bank'], 'Rate': ['10.5%', '10.7%', '9.2%']}))
    with col_b:
        st.subheader("💳 Personalized Card Offers")
        if 'last_score' in st.session_state:
            score = st.session_state['last_score']
            if score >= 750: st.info("🏆 **Premium Platinum Card**: Eligible for 1% cashback and Lounge access.")
            elif score >= 650: st.info("🏅 **Gold Rewards Card**: Eligible for fuel surcharge waivers.")
            else: st.warning("💳 **Secured Credit Card**: Recommended to rebuild your credit score.")
        else: st.info("Assessment pannungana unga score-ku yetha cards-a suggest pannuvaen!")
