import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import io

# 1. Page Configuration
st.set_page_config(page_title="Loan Intelligence AI | Hari Murugan", layout="wide")

# UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #00CC96; color: white; border-radius: 8px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('loan_model_pipeline.sav')

model = load_model()

# --- HELPER: PDF GENERATION ---
def generate_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="System Audit Logs - Loan Intelligence AI", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    
    # Header
    pdf.set_fill_color(200, 220, 255)
    cols = ["Date", "Name", "Income", "Score", "Result"]
    for col in cols:
        pdf.cell(38, 10, col, 1, 0, 'C', 1)
    pdf.ln()
    
    # Rows (Taking last 20 records to keep PDF clean)
    for index, row in df.tail(20).iterrows():
        pdf.cell(38, 10, str(row['Timestamp'])[:10], 1)
        pdf.cell(38, 10, str(row['Applicant_Name'])[:15], 1)
        pdf.cell(38, 10, str(row['Annual_Income']), 1)
        pdf.cell(38, 10, str(row['Credit_Score']), 1)
        pdf.cell(38, 10, str(row['Prediction']), 1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin-1')

# --- PREVIOUS FEATURES (Assessment, XAI, Market) ---
# [Keeping all Tab 1, 2, 3, 4 logic as same as before]

# --- TAB 5: ADMIN CENTER (Updated with Download Options) ---
tabs = st.tabs(["👤 Assessment", "📂 Bulk Processing", "📈 Explainable AI", "🏦 Market Rates", "🔐 Admin Center"])

with tabs[4]:
    st.header("🔐 Admin Data Center")
    st.write("**DEV:** Hari murugan | Data Scientist")
    
    pwd = st.text_input("Admin Password", type="password")
    if pwd == "admin123":
        if os.path.exists('user_logs.csv'):
            df_logs = pd.read_csv('user_logs.csv')
            st.dataframe(df_logs.tail(10), use_container_width=True)
            
            st.markdown("### 📥 Export System Logs")
            c1, c2 = st.columns(2)
            
            # Format Selection
            format_choice = c1.selectbox("Select Export Format", ["CSV (Excel Compatible)", "PDF (Official Report)"])
            
            if format_choice == "CSV (Excel Compatible)":
                csv = df_logs.to_csv(index=False).encode('utf-8')
                c2.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"Hari_Murugan_Logs_{datetime.date.today()}.csv",
                    mime="text/csv"
                )
            
            elif format_choice == "PDF (Official Report)":
                pdf_data = generate_pdf_report(df_logs)
                c2.download_button(
                    label="Download as PDF",
                    data=pdf_data,
                    file_name=f"System_Audit_{datetime.date.today()}.pdf",
                    mime="application/pdf"
                )
                
            if st.button("🗑️ Clear Log History"):
                os.remove('user_logs.csv')
                st.success("Logs cleared!")
                st.rerun()
        else:
            st.info("No logs found yet.")
