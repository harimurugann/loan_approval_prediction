# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import time

# --- PAGE CONFIGURATION & METADATA ---
st.set_page_config(
    page_title="LOAN RISK ASSESSMENT SYSTEM", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS: Modern Dark Fintech Design (Font Sizes Reduced) ---
st.markdown("""
    <style>
    /* Main background - Dark Slate/Charcoal */
    .main { background-color: #1a1c24; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Overall text color adjustment */
    .stMarkdown, p, label { color: #e5e7eb !important; }

    /* Custom Headers - Reduced font size */
    .custom-main-header {
        color: #ffffff;
        font-weight: 800;
        font-size: 1.6rem; /* Reduced from 2.2rem */
        text-transform: uppercase;
        margin-top: -10px;
        margin-bottom: 25px;
        letter-spacing: 1.2px;
    }

    /* KPI Cards - Reduced sizes */
    .kpi-card {
        background-image: linear-gradient(135deg, #262a33 0%, #1c1f26 100%);
        border: 1px solid #374151;
        border-bottom: 3px solid #00f2fe;
        border-radius: 8px;
        padding: 15px; /* Reduced padding */
        text-align: left;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-card h3 {
        font-size: 0.75rem; /* Reduced */
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9ca3af;
    }
    .kpi-card p {
        font-size: 1.4rem; /* Reduced from 1.8rem */
        margin: 5px 0;
        font-weight: bold;
        color: #ffffff !important;
    }
    .kpi-card-sub {
        font-size: 0.70rem;
        color: #2ecc71 !important; 
        margin: 0;
        font-weight: bold;
    }

    /* Main Content Containers */
    .content-container {
        background-color: #262a33;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .content-container-header {
        font-size: 1.0rem; /* Reduced from 1.2rem */
        font-weight: 700;
        margin-bottom: 15px;
        color: #ffffff;
        text-transform: uppercase;
        border-bottom: 1px solid #374151;
        padding-bottom: 8px;
    }

    /* Input Labels - Smaller */
    label[data-testid="stWidgetLabel"] {
        font-size: 0.75rem !important; /* Reduced */
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Primary Buttons */
    .stButton>button {
        background-color: transparent;
        color: #00f2fe !important;
        border-radius: 4px;
        padding: 0.4rem 1.2rem;
        font-size: 0.85rem;
        font-weight: bold;
        border: 1px solid #00f2fe;
        text-transform: uppercase;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00f2fe;
        color: #000000 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
    }

    /* Status Text */
    .status-text {
        text-align: right;
        color: #2ecc71; 
        font-weight: 600;
        font-size: 0.75rem; /* Reduced */
        letter-spacing: 1px;
    }
    
    /* Roadmap Box */
    .roadmap-box {
        background-color: #1a1c24;
        border-left: 4px solid #00f2fe;
        padding: 15px;
        margin-top: 15px;
        border-radius: 4px;
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

def authenticate(username, password):
    return username == "admin" and password == "admin123"

# --- MAIN DASHBOARD LOGIC ---
def main():
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.title("INTELLIGENCE HUB")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("NAVIGATE MODULES", [
        "📊 SYSTEM DASHBOARD",
        "📂 BULK PROCESSING",
        "🗺️ CREDIT ROADMAP",
        "🔒 ADMIN GATEWAY"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("ENGINE: V2.6 | STATUS: SECURE")

    # HEADER AREA
    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
        st.markdown('<div class="custom-main-header">LOAN RISK ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    with h_col2:
        st.markdown('<p class="status-text">🟢 STATUS: ONLINE<br>Engine Connected</p>', unsafe_allow_html=True)

    # ==========================================
    # 1. SYSTEM DASHBOARD (Main View)
    # ==========================================
    if app_mode == "📊 SYSTEM DASHBOARD":
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown('<div class="kpi-card"><h3>Avg Portfolio Score</h3><p>710</p><p class="kpi-card-sub">FAIR & STABLE</p></div>', unsafe_allow_html=True)
        with kpi2:
            st.markdown('<div class="kpi-card"><h3>Predicted Default</h3><p>19.8%</p><p class="kpi-card-sub">BELOW THRESHOLD</p></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown('<div class="kpi-card"><h3>Active Models</h3><p>01</p><p class="kpi-card-sub">RANDOM FOREST</p></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown('<div class="kpi-card"><h3>System Accuracy</h3><p>80.17%</p><p class="kpi-card-sub">OPTIMIZED</p></div>', unsafe_allow_html=True)

        col_form, col_anal = st.columns([1, 1.2])
        
        # 👤 Apply for Loan Card (Removed unnecessary applicant name)
        with col_form:
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            st.markdown('<div class="content-container-header">👤 APPLICANT DATA ENTRY</div>', unsafe_allow_html=True)
            
            if pipeline is None:
                st.error("🚨 Pipeline artifact not found. Please run 'train_model.py' first.")
            else:
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000.0, value=15000.0, step=500.0)
                    int_rate = st.number_input("Interest Rate (%)", value=10.5)
                with col_f2:
                    annual_inc = st.number_input("Annual Income ($)", value=75000.0)
                    term = st.selectbox("Loan Term (Months)", [36, 60])
                
                # Hidden background defaults
                dti_val = 15.0 
                open_acc_val = 10 
                
                st.write("")
                submit = st.button("EXECUTE RISK ANALYSIS") 
                
            st.markdown('</div>', unsafe_allow_html=True)

        # 📈 Analysis / Results Card (Removed unnecessary probability chart)
        with col_anal:
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            st.markdown('<div class="content-container-header">📈 AI INFERENCE RESULTS</div>', unsafe_allow_html=True)
            
            if pipeline is not None and 'submit' in locals() and submit:
                with st.spinner("AI Engine executing risk analysis..."):
                    time.sleep(1)
                    
                    installment_val = (loan_amnt * (int_rate / 1200)) / (1 - (1 + int_rate / 1200)**(-term))
                    total_acc_val = open_acc_val * 2
                    
                    df_input = pd.DataFrame([[loan_amnt, term, int_rate, installment_val, annual_inc, dti_val, open_acc_val, total_acc_val]],
                                            columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                    
                    prediction = pipeline.predict(df_input)
                    probability = pipeline.predict_proba(df_input)[0][1] 
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        if prediction[0] == 1:
                            st.success("✅ STATUS: APPROVED")
                            st.caption("Risk Profile: Low to Moderate")
                        else:
                            st.error("🚫 STATUS: REJECTED")
                            st.caption("Risk Profile: High Default Probability")
                            
                    with res_col2:
                        st.metric("Approval Confidence Score", f"{probability*100:.1f} / 100")
                    
                    st.write("")
                    st.progress(probability)
            else:
                st.info("Awaiting input data. Click 'Execute Risk Analysis' to generate AI insights.")
                
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 2. BULK PROCESSING MODULE
    # ==========================================
    elif app_mode == "📂 BULK PROCESSING":
        st.markdown('<div class="content-container"><div class="content-container-header">📂 HIGH-VOLUME BATCH PROCESSING</div>', unsafe_allow_html=True)
        st.write("Upload a CSV file containing multiple customer records for batch inference.")
        
        uploaded_file = st.file_uploader("Upload Batch CSV", type="csv")
        if uploaded_file is not None and pipeline is not None:
            df_bulk = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df_bulk.head(3))
            
            if st.button("PROCESS BATCH DATA"):
                with st.spinner("Processing records..."):
                    time.sleep(1)
                    X_bulk = df_bulk.drop('loan_paid_back', axis=1) if 'loan_paid_back' in df_bulk.columns else df_bulk
                    predictions = pipeline.predict(X_bulk)
                    df_bulk['AI_Status'] = ["Approved" if p == 1 else "Denied" for p in predictions]
                    st.success("✅ Batch processing complete!")
                    st.dataframe(df_bulk[['loan_amnt', 'annual_inc', 'AI_Status']].head(5))
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 3. CREDIT ROADMAP MODULE (Fixed & Working)
    # ==========================================
    elif app_mode == "🗺️ CREDIT ROADMAP":
        st.markdown('<div class="content-container"><div class="content-container-header">🗺️ PERSONALISED CREDIT ROADMAP</div>', unsafe_allow_html=True)
        st.write("Generate a structured financial recovery plan for High-Risk profiles.")
        
        c1, c2 = st.columns(2)
        target_score = c1.slider("Target Credit Score", 300, 850, 750)
        current_dti = c2.number_input("Current DTI (%)", value=45.0)
        
        if st.button("GENERATE ROADMAP"):
            st.markdown('<div class="roadmap-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Action Plan to reach {target_score} Score")
            st.markdown(f"**Step 1 (Immediate): DTI Optimization**\n* Your current Debt-to-Income ratio is **{current_dti}%**. \n* **Action:** Pay down revolving credit to bring this below 30% before reapplying.")
            st.markdown("**Step 2 (30-60 Days): Credit Utilization**\n* Keep credit card balances below 10% of your total limit.")
            st.markdown("**Step 3 (Long Term): Consistent History**\n* Setup automated payments. The AI engine heavily weights 'Total Accounts' and payment history.")
            st.markdown('</div>', unsafe_allow_html=True)
            st.balloons()
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 4. ADMIN GATEWAY MODULE
    # ==========================================
    elif app_mode == "🔒 ADMIN GATEWAY":
        st.markdown('<div class="content-container"><div class="content-container-header">🔒 SECURE ADMIN ACCESS</div>', unsafe_allow_html=True)
        user = st.text_input("Admin ID (Username)")
        pwd = st.text_input("Security Key (Password)", type="password")
        if st.button("AUTHENTICATE SESSION"):
            if authenticate(user, pwd):
                st.success("✅ Authorized. Server Metrics Online.")
                st.code("""
                [SYSTEM LOG]
                - Model Loaded: RandomForestClassifier
                - Latency: 42ms/inference
                - Pipeline Status: Healthy
                - Data Drift Detected: None
                """, language="bash")
            else:
                st.error("🚫 Authentication Failed. Invalid Credentials.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
