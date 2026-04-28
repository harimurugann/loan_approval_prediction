# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import time
import random

# --- PAGE CONFIGURATION & METADATA ---
st.set_page_config(
    page_title="LOAN RISK ASSESSMENT SYSTEM", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS: Modern Dark Fintech Design ---
st.markdown("""
    <style>
    .main { background-color: #1a1c24; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stMarkdown, p, label { color: #e5e7eb !important; }
    .custom-main-header {
        color: #ffffff; font-weight: 800; font-size: 1.6rem; text-transform: uppercase;
        margin-top: -10px; margin-bottom: 25px; letter-spacing: 1.2px;
    }
    .kpi-card {
        background-image: linear-gradient(135deg, #262a33 0%, #1c1f26 100%);
        border: 1px solid #374151; border-bottom: 3px solid #00f2fe;
        border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-card h3 { font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; }
    .kpi-card p { font-size: 1.4rem; margin: 5px 0; font-weight: bold; color: #ffffff !important; }
    .kpi-card-sub { font-size: 0.70rem; color: #2ecc71 !important; margin: 0; font-weight: bold; }
    .content-container {
        background-color: #262a33; border-radius: 12px; padding: 20px;
        margin-bottom: 20px; border: 1px solid #374151; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .content-container-header {
        font-size: 1.0rem; font-weight: 700; margin-bottom: 15px; color: #ffffff;
        text-transform: uppercase; border-bottom: 1px solid #374151; padding-bottom: 8px;
    }
    label[data-testid="stWidgetLabel"] { font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
    .stButton>button, .stDownloadButton>button {
        background-color: transparent; color: #00f2fe !important; border-radius: 4px;
        padding: 0.4rem 1.2rem; font-size: 0.85rem; font-weight: bold;
        border: 1px solid #00f2fe; text-transform: uppercase; transition: 0.3s; width: 100%;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background-color: #00f2fe; color: #000000 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; }
    .status-text { text-align: right; color: #2ecc71; font-weight: 600; font-size: 0.75rem; letter-spacing: 1px; }
    .roadmap-box { background-color: #1a1c24; border-left: 4px solid #00f2fe; padding: 15px; margin-top: 15px; border-radius: 4px; }
    .stream-active { color: #00f2fe; font-weight: bold; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
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
        "🌐 LIVE API STREAM",
        "📂 BULK PROCESSING",
        "🗺️ CREDIT ROADMAP",
        "🔒 ADMIN GATEWAY"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("ENGINE: V3.0 | STATUS: SECURE")

    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
        st.markdown('<div class="custom-main-header">LOAN RISK ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    with h_col2:
        st.markdown('<p class="status-text">🟢 STATUS: ONLINE<br>Engine Connected</p>', unsafe_allow_html=True)

    # ==========================================
    # 1. SYSTEM DASHBOARD
    # ==========================================
    if app_mode == "📊 SYSTEM DASHBOARD":
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1: st.markdown('<div class="kpi-card"><h3>Avg Portfolio Score</h3><p>710</p><p class="kpi-card-sub">FAIR & STABLE</p></div>', unsafe_allow_html=True)
        with kpi2: st.markdown('<div class="kpi-card"><h3>Predicted Default</h3><p>19.8%</p><p class="kpi-card-sub">BELOW THRESHOLD</p></div>', unsafe_allow_html=True)
        with kpi3: st.markdown('<div class="kpi-card"><h3>Active Models</h3><p>01</p><p class="kpi-card-sub">RANDOM FOREST</p></div>', unsafe_allow_html=True)
        with kpi4: st.markdown('<div class="kpi-card"><h3>System Accuracy</h3><p>80.17%</p><p class="kpi-card-sub">OPTIMIZED</p></div>', unsafe_allow_html=True)

        col_form, col_anal = st.columns([1, 1.2])
        with col_form:
            st.markdown('<div class="content-container"><div class="content-container-header">👤 APPLICANT DATA ENTRY</div>', unsafe_allow_html=True)
            if pipeline is None:
                st.markdown("<p style='color:#e74c3c;'>🚨 Pipeline not found. Run 'train_model.py' first.</p>", unsafe_allow_html=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    loan_amnt = st.number_input("Loan Amount ($)", 1000.0, value=15000.0, step=500.0)
                    int_rate = st.number_input("Interest Rate (%)", value=10.5)
                with c2:
                    annual_inc = st.number_input("Annual Income ($)", value=75000.0)
                    term = st.selectbox("Loan Term (Months)", [36, 60])
                submit = st.button("EXECUTE RISK ANALYSIS") 
            st.markdown('</div>', unsafe_allow_html=True)

        with col_anal:
            st.markdown('<div class="content-container"><div class="content-container-header">📈 AI INFERENCE RESULTS</div>', unsafe_allow_html=True)
            if pipeline is not None and 'submit' in locals() and submit:
                with st.spinner("Executing risk analysis..."):
                    time.sleep(1)
                    installment_val = (loan_amnt * (int_rate / 1200)) / (1 - (1 + int_rate / 1200)**(-term))
                    df_input = pd.DataFrame([[loan_amnt, term, int_rate, installment_val, annual_inc, 15.0, 10, 20]],
                                            columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                    prediction = pipeline.predict(df_input)
                    probability = pipeline.predict_proba(df_input)[0][1] 
                    
                    r1, r2 = st.columns(2)
                    with r1:
                        if prediction[0] == 1:
                            st.markdown("<h3 style='color:#2ecc71; margin-bottom: 0px;'>✅ APPROVED</h3><p style='color:#9ca3af; font-size: 0.85rem;'>Risk Profile: Low to Moderate</p>", unsafe_allow_html=True)
                        else:
                            st.markdown("<h3 style='color:#e74c3c; margin-bottom: 0px;'>🚫 REJECTED</h3><p style='color:#9ca3af; font-size: 0.85rem;'>Risk Profile: High Default Probability</p>", unsafe_allow_html=True)
                    with r2: st.metric("Confidence Score", f"{probability*100:.1f} / 100")
            else:
                st.markdown("<p style='color:#9ca3af; font-style:italic;'>Awaiting input data. Click 'Execute' to generate AI insights.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 2. LIVE API STREAM SIMULATION (UPDATED FOR TESTING DATA)
    # ==========================================
    elif app_mode == "🌐 LIVE API STREAM":
        st.markdown('<div class="content-container"><div class="content-container-header">📡 REAL-TIME API STREAM INFERENCE</div>', unsafe_allow_html=True)
        st.write("Upload a testing dataset (CSV). The system will simulate streaming this data row-by-row through the ML API.")

        # File uploader for Testing Data
        stream_file = st.file_uploader("Upload Testing Data for Stream (CSV)", type="csv")
        
        if 'stream_active' not in st.session_state:
            st.session_state.stream_active = False

        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            if st.button("▶️ START STREAM"):
                st.session_state.stream_active = True
        with c2:
            if st.button("⏹️ STOP STREAM"):
                st.session_state.stream_active = False

        if st.session_state.stream_active:
            st.markdown("<p class='stream-active'>🟢 System is actively streaming and processing records...</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#9ca3af;'>🔴 Stream is currently offline.</p>", unsafe_allow_html=True)

        st.markdown("---")
        
        placeholder = st.empty()

        if st.session_state.stream_active and pipeline is not None:
            if 'live_df' not in st.session_state:
                st.session_state.live_df = pd.DataFrame(columns=["Timestamp", "App_ID", "Req_Amount", "Income", "AI_Decision", "Confidence"])

            with placeholder.container():
                st.write("Live Data Feed:")
                table_placeholder = st.empty()

            # Check if user uploaded a file
            if stream_file is not None:
                df_test = pd.read_csv(stream_file)
                
                # Stream row by row from the uploaded CSV
                for index, row in df_test.iterrows():
                    if not st.session_state.stream_active:
                        break
                    
                    try:
                        app_id = f"APP-{random.randint(10000, 99999)}"
                        l_amnt = row['loan_amnt']
                        a_inc = row['annual_inc']
                        
                        # Prepare row for prediction (matching pipeline columns)
                        df_stream = pd.DataFrame([row])
                        if 'loan_paid_back' in df_stream.columns:
                            df_stream = df_stream.drop('loan_paid_back', axis=1)
                            
                        pred = pipeline.predict(df_stream)
                        prob = pipeline.predict_proba(df_stream)[0][1]
                        decision = "✅ APPROVED" if pred[0] == 1 else "🚫 REJECTED"
                        
                        new_record = pd.DataFrame({
                            "Timestamp": [time.strftime("%H:%M:%S")],
                            "App_ID": [app_id],
                            "Req_Amount": [f"${l_amnt:,.0f}"],
                            "Income": [f"${a_inc:,.0f}"],
                            "AI_Decision": [decision],
                            "Confidence": [f"{prob*100:.1f}%"]
                        })
                        
                        st.session_state.live_df = pd.concat([new_record, st.session_state.live_df]).head(10)
                        table_placeholder.dataframe(st.session_state.live_df, use_container_width=True)
                        time.sleep(1.2) # API network delay simulation
                        
                    except Exception as e:
                        st.error(f"Data format error in row {index}. Make sure columns match training data.")
                        break
            else:
                # Fallback: Auto-generate if no file is uploaded
                st.info("No file uploaded. Generating synthetic test data for stream...")
                for i in range(20):
                    if not st.session_state.stream_active:
                        break
                    app_id = f"APP-{random.randint(10000, 99999)}"
                    l_amnt = random.uniform(2000, 40000)
                    term = random.choice([36, 60])
                    i_rate = random.uniform(5.0, 25.0)
                    a_inc = random.uniform(30000, 150000)
                    dti = random.uniform(5.0, 35.0)
                    inst = (l_amnt * (i_rate / 1200)) / (1 - (1 + i_rate / 1200)**(-term))
                    
                    df_stream = pd.DataFrame([[l_amnt, term, i_rate, inst, a_inc, dti, 10, 20]],
                                            columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                    
                    pred = pipeline.predict(df_stream)
                    prob = pipeline.predict_proba(df_stream)[0][1]
                    decision = "✅ APPROVED" if pred[0] == 1 else "🚫 REJECTED"
                    
                    new_record = pd.DataFrame({
                        "Timestamp": [time.strftime("%H:%M:%S")],
                        "App_ID": [app_id],
                        "Req_Amount": [f"${l_amnt:,.0f}"],
                        "Income": [f"${a_inc:,.0f}"],
                        "AI_Decision": [decision],
                        "Confidence": [f"{prob*100:.1f}%"]
                    })
                    
                    st.session_state.live_df = pd.concat([new_record, st.session_state.live_df]).head(10)
                    table_placeholder.dataframe(st.session_state.live_df, use_container_width=True)
                    time.sleep(1.2)

            st.session_state.stream_active = False
            st.success("✅ Stream Session Completed.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 3. BULK PROCESSING
    # ==========================================
    elif app_mode == "📂 BULK PROCESSING":
        st.markdown('<div class="content-container"><div class="content-container-header">📂 HIGH-VOLUME BATCH PROCESSING</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Batch CSV", type="csv")
        if uploaded_file is not None and pipeline is not None:
            df_bulk = pd.read_csv(uploaded_file)
            if st.button("PROCESS BATCH DATA"):
                X_bulk = df_bulk.drop('loan_paid_back', axis=1) if 'loan_paid_back' in df_bulk.columns else df_bulk
                predictions = pipeline.predict(X_bulk)
                df_bulk['AI_Status'] = ["Approved" if p == 1 else "Denied" for p in predictions]
                st.markdown("<h4 style='color:#2ecc71;'>✅ Batch processing complete!</h4>", unsafe_allow_html=True)
                st.dataframe(df_bulk[['loan_amnt', 'annual_inc', 'AI_Status']].head(5))
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 4. CREDIT ROADMAP
    # ==========================================
    elif app_mode == "🗺️ CREDIT ROADMAP":
        st.markdown('<div class="content-container"><div class="content-container-header">🗺️ PERSONALISED CREDIT ROADMAP</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        target_score = c1.slider("Target Credit Score", 300, 850, 750)
        current_dti = c2.number_input("Current DTI (%)", value=45.0)
        if st.button("GENERATE ROADMAP"):
            st.markdown('<div class="roadmap-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Action Plan to reach {target_score} Score")
            st.markdown(f"**Step 1 (Immediate):** Pay down revolving credit to bring {current_dti}% DTI below 30%.")
            st.markdown("**Step 2 (30-60 Days):** Keep credit card balances below 10% of total limit.")
            st.markdown("**Step 3 (Long Term):** Setup automated payments for consistent history.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 5. ADMIN GATEWAY
    # ==========================================
    elif app_mode == "🔒 ADMIN GATEWAY":
        st.markdown('<div class="content-container"><div class="content-container-header">🔒 SECURE ADMIN ACCESS</div>', unsafe_allow_html=True)
        user = st.text_input("Admin ID")
        pwd = st.text_input("Security Key", type="password")
        if st.button("AUTHENTICATE SESSION"):
            if authenticate(user, pwd):
                st.markdown("<h4 style='color:#2ecc71;'>✅ Authorized. Server Metrics Online.</h4>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#e74c3c;'>🚫 Authentication Failed.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
