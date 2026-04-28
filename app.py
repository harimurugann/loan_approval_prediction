# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import time
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Loan Risk AI", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DARK UI DESIGN & ROADMAP ---
st.markdown("""
    <style>
    /* Dark background for the main app to make white text visible */
    .main { background-color: #121212; color: #ffffff; }
    
    /* Global text color adjustment for dark mode */
    .stMarkdown, p, label { color: #e5e7eb !important; }
    
    /* Button styling */
    .stButton>button {
        background-color: #00b894;
        color: #000000 !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #55efc4;
        transform: scale(1.02);
    }
    
    /* Custom Headers for Dark Mode */
    .custom-header {
        color: #ffffff;
        font-weight: 800;
        font-size: 2rem;
        border-bottom: 2px solid #374151;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Black Background Box specifically for Credit Roadmap */
    .roadmap-container {
        background-color: #000000;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
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
    # Sidebar Navigation
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2602/2602112.png", width=80)
    st.sidebar.title("Risk Intelligence Hub")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("Navigate Modules", [
        "📊 Live Dashboard & Analytics",
        "👤 Single Risk Assessment",
        "📂 Bulk Batch Processing",
        "🗺️ Personalised Credit Roadmap",
        "🔒 Secure Admin Gateway"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("System v2.1.0 | Dark Mode Edition")

    # ==========================================
    # MODULE 1: LIVE DASHBOARD
    # ==========================================
    if app_mode == "📊 Live Dashboard & Analytics":
        st.markdown('<div class="custom-header">📊 Data Analytics & Portfolio Overview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Total Active Loans Models", value="1", delta="Optimized")
        col2.metric(label="System Accuracy", value="80.17%", delta="Stable")
        col3.metric(label="Processing Engine", value="Pipeline Active", delta="Online")

    # ==========================================
    # MODULE 2: SINGLE RISK ASSESSMENT
    # ==========================================
    elif app_mode == "👤 Single Risk Assessment":
        st.markdown('<div class="custom-header">👤 AI Underwriting Engine</div>', unsafe_allow_html=True)
        # (Assuming you keep the same inputs as previous code for brevity, you can paste the form here)
        st.info("Input fields will match the dark theme automatically.")

    # ==========================================
    # MODULE 13: PERSONALISED CREDIT ROADMAP (NEW & STYLED)
    # ==========================================
    elif app_mode == "🗺️ Personalised Credit Roadmap":
        st.markdown('<div class="custom-header">🗺️ Personalised Credit Roadmap</div>', unsafe_allow_html=True)
        
        st.write("Generate a custom financial improvement plan for clients based on their current metrics.")
        
        col1, col2 = st.columns(2)
        target_score = col1.slider("Target Credit Score", 300, 850, 750)
        current_dti = col2.number_input("Current DTI (%)", value=35.0)
        
        if st.button("Generate Custom Roadmap"):
            # The black background container starts here
            st.markdown('<div class="roadmap-container">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 Roadmap to Score: {target_score}")
            st.markdown("---")
            st.markdown("#### 🛠️ Immediate Actions (0-30 Days)")
            st.markdown(f"1. **DTI Optimization:** Your current DTI is {current_dti}%. Aim to reduce this below 30% by paying off high-interest revolving credit.\n"
                        "2. **Credit Utilization:** Keep credit card balances below 10% of your total limit.\n"
                        "3. **Dispute Inaccuracies:** Pull a fresh credit report and flag any hard inquiries you don't recognize.")
            
            st.markdown("#### 📈 Mid-Term Goals (3-6 Months)")
            st.markdown("1. **Consistent Payments:** Setup autopay for all installments to ensure zero late marks.\n"
                        "2. **Credit Mix:** If you only have credit cards, consider a small, secure credit-builder loan to diversify your portfolio.")
            
            st.markdown("#### 🚀 Long-Term Strategy (6+ Months)")
            st.markdown("1. **Aged Accounts:** Do not close your oldest credit accounts, even if unused. Account age heavily impacts your score.\n"
                        "2. **Limit Increases:** Request a credit limit increase on existing cards without triggering a hard pull.")
            
            st.markdown('</div>', unsafe_allow_html=True) # Container ends here

    # ==========================================
    # MODULE 4: SECURE ADMIN GATEWAY
    # ==========================================
    elif app_mode == "🔒 Secure Admin Gateway":
        st.markdown('<div class="custom-header">🔒 Level 4 Restricted Access</div>', unsafe_allow_html=True)
        user = st.text_input("Admin ID (Username)")
        pwd = st.text_input("Security Key (Password)", type="password")
        if st.button("🔐 Authenticate Session"):
            if authenticate(user, pwd):
                st.success("✅ Access Granted.")
            else:
                st.error("🚫 Authentication Failed.")

if __name__ == "__main__":
    main()
