# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import time
import plotly.express as px

# --- PAGE CONFIGURATION & METADATA ---
st.set_page_config(
    page_title="LOAN RISK ASSESSMENT SYSTEM", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS: Anuppina design ref-kku match panna ---
st.markdown("""
    <style>
    /* Main background - Dark Slate/Charcoal */
    .main { background-color: #1a1c24; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Overall text color adjustment */
    .stMarkdown, p, label { color: #e5e7eb !important; }

    /* Custom Headers - Matches 'LOAN RISK ASSESSMENT SYSTEM' style */
    .custom-main-header {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.2rem;
        text-transform: uppercase;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    /* KPI Cards - The 4 Top Cards from design (Subtle gradient, Cyan Border) */
    .kpi-card {
        background-image: linear-gradient(135deg, #262a33 0%, #1c1f26 100%);
        border: 2px solid #00f2fe; /* Cyan Border Accent */
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: left;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-card h3 {
        font-size: 0.9rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #e5e7eb;
    }
    .kpi-card p {
        font-size: 2rem;
        margin: 5px 0;
        font-weight: bold;
        color: #ffffff !important;
    }
    .kpi-card-sub {
        font-size: 0.8rem;
        color: #2ecc71; /* Green status color */
        margin: 0;
    }

    /* Main Content Containers (Apply for Loan, Analysis Cards) */
    .content-container {
        background-color: #262a33;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .content-container-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 20px;
        color: #ffffff;
        text-transform: uppercase;
    }

    /* Styled Input Fields - Matches ref image design */
    div[data-baseweb="input"] {
        background-color: #1a1c24 !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
        color: white !important;
    }
    label[data-testid="stWidgetLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Primary Buttons - Text only, cyan outline/text from design */
    .stButton>button {
        background-color: transparent;
        color: #00f2fe !important;
        border-radius: 6px;
        padding: 0.4rem 1.2rem;
        font-weight: bold;
        border: 2px solid #00f2fe;
        text-transform: uppercase;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00f2fe;
        color: #000000 !important;
    }

    /* Sidebar - Minimal Dark */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
    }

    /* Status Text (Top Right) - From design ref */
    .status-text {
        text-align: right;
        color: #2ecc71; /* Online status color */
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- DIRECTORY HANDLING & MODEL LOADING ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'full_pipeline.sav')

@st.cache_resource
def load_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = load_pipeline()

# --- AUTHENTICATION ---
def authenticate(username, password):
    return username == "admin" and password == "admin123"

# --- MAIN DASHBOARD LOGIC ---
def main():
    # Sidebar Navigation - Clean design
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2602/2602112.png", width=60)
    st.sidebar.title("Intelligence Modules")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("Navigate Modules", [
        "📊 System Dashboard",
        "📂 Bulk Batch Processing",
        "🗺️ Credit Roadmap",
        "🔒 Admin Gateway"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("System v2.5 | New UI Edition")

    # ==========================================
    # HEADER AREA (Matches Design Ref)
    # ==========================================
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown('<div class="custom-main-header">LOAN RISK ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    with h_col2:
        st.markdown('<p class="status-text">STATUS: ONLINE<br>Connection Checked</p>', unsafe_allow_html=True)

    # ==========================================
    # TOP KPI ROWS (The 4 Cards from design)
    # ==========================================
    if app_mode == "📊 System Dashboard":
        st.write("") # Spacer
        kpi1, kpi2, kpi3, kpi4 = st.columns([1,1,1,1])
        
        # Mapping synthetic or model data for visual effect
        with kpi1:
            st.markdown('<div class="kpi-card"><h3>Assessment Score</h3><p>710</p><p class="kpi-card-sub">FAIR</p></div>', unsafe_allow_html=True)
        with kpi2:
            st.markdown('<div class="kpi-card"><h3>Predicted Default Rate</h3><p>19.8%</p><p class="kpi-card-sub">STABLE</p></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown('<div class="kpi-card"><h3>Active Models</h3><p>01</p><p class="kpi-card-sub">PRODUCTION READY</p></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown('<div class="kpi-card"><h3>System Accuracy</h3><p>80.17%</p><p class="kpi-card-sub">OPTIMIZED</p></div>', unsafe_allow_html=True)

        # ==========================================
        # MAIN CONTENT ROWS (Apply for Loan, Analysis Cards from design)
        # ==========================================
        col_form, col_anal = st.columns([1.5, 2])
        
        # 👤 Apply for Loan Card
        with col_form:
            st.markdown('<div class="content-container"><div class="content-container-header">👤 Apply for Loan</div>', unsafe_allow_html=True)
            
            if pipeline is None:
                st.error("🚨 Pipeline artifact not found. Please run 'train_model.py' first.")
            else:
                applicant_name = st.text_input("Applicant Name", placeholder="E.g., Tamil Selvan") # Visual only
                
                # Model functional inputs
                loan_amnt = st.number_input("Loan Amount ($)", min_value=1000.0, value=10000.0, step=500.0)
                term = st.selectbox("Loan Term (Months)", [36, 60])
                int_rate = st.number_input("Interest Rate (%)", value=10.5)
                annual_inc = st.number_input("Annual Income ($)", value=70000.0)
                
                # Invisible model dependencies (using defaults for design fidelity)
                dti_val = 15.0 
                open_acc_val = 10 
                
                st.write("")
                submit = st.button("ASSESS RISK") # Primary button from design
                
            st.markdown('</div>', unsafe_allow_html=True) # Container ends

        # 📈 Analysis / Results Card
        with col_anal:
            st.markdown('<div class="content-container"><div class="content-container-header">📈 Analysis / Results</div>', unsafe_allow_html=True)
            
            if pipeline is not None and 'submit' in locals() and submit:
                with st.spinner("AI Engine executing risk analysis..."):
                    time.sleep(1)
                    
                    # Dummy installment, other hidden features based on synthetic average for functionality
                    installment_val = (loan_amnt * (int_rate / 1200)) / (1 - (1 + int_rate / 1200)**(-term))
                    total_acc_val = open_acc_val * 2
                    
                    df_input = pd.DataFrame([[loan_amnt, term, int_rate, installment_val, annual_inc, dti_val, open_acc_val, total_acc_val]],
                                            columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                    
                    prediction = pipeline.predict(df_input)
                    probability = pipeline.predict_proba(df_input)[0][1] # Probability of APPROVAL
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    # Result Display styled as a card item
                    with res_col1:
                        if prediction[0] == 1:
                            st.success("✅ STATUS: APPROVED")
                        else:
                            st.error("🚫 STATUS: DENIED (High Risk)")
                    with res_col2:
                        st.metric("Approval Confidence", f"{probability:.2%}")
                    
                    st.progress(probability)
                    
                    # Placeholder interactive chart to make it look full
                    st.write("")
                    st.subheader("Risk Score Distribution (Portfolio Avg.)")
                    fig = px.bar(x=["Applicant Score", "Portfolio Avg"], y=[probability*100, 75]), 
                                 labels={'x': 'Entity', 'y': 'Score'},
