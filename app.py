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

# --- CUSTOM CSS: Modern Dark Fintech Design ---
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
        letter-spacing: 1.5px;
    }

    /* KPI Cards - The 4 Top Cards from design (Subtle gradient, Cyan Border) */
    .kpi-card {
        background-image: linear-gradient(135deg, #262a33 0%, #1c1f26 100%);
        border: 1px solid #374151;
        border-bottom: 3px solid #00f2fe; /* Cyan Border Accent */
        border-radius: 8px;
        padding: 20px;
        text-align: left;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-card h3 {
        font-size: 0.85rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9ca3af;
    }
    .kpi-card p {
        font-size: 1.8rem;
        margin: 5px 0;
        font-weight: bold;
        color: #ffffff !important;
    }
    .kpi-card-sub {
        font-size: 0.75rem;
        color: #2ecc71 !important; /* Green status color */
        margin: 0;
        font-weight: bold;
    }

    /* Main Content Containers (Apply for Loan, Analysis Cards) */
    .content-container {
        background-color: #262a33;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .content-container-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 20px;
        color: #ffffff;
        text-transform: uppercase;
        border-bottom: 1px solid #374151;
        padding-bottom: 10px;
    }

    /* Primary Buttons - Text only, cyan outline/text from design */
    .stButton>button {
        background-color: transparent;
        color: #00f2fe !important;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
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

    /* Sidebar - Minimal Dark */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
    }

    /* Status Text (Top Right) - From design ref */
    .status-text {
        text-align: right;
        color: #2ecc71; /* Online status color */
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 1px;
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
    st.sidebar.title("INTELLIGENCE HUB")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("NAVIGATE MODULES", [
        "📊 SYSTEM DASHBOARD",
        "📂 BULK PROCESSING",
        "🗺️ CREDIT ROADMAP",
        "🔒 ADMIN GATEWAY"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("ENGINE: V2.5 | STATUS: SECURE")

    # ==========================================
    # HEADER AREA (Matches Design Ref)
    # ==========================================
    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
        st.markdown('<div class="custom-main-header">LOAN RISK ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    with h_col2:
        st.markdown('<p class="status-text">🟢 STATUS: ONLINE<br>Engine Connected</p>', unsafe_allow_html=True)

    # ==========================================
    # SYSTEM DASHBOARD (Main View)
    # ==========================================
    if app_mode == "📊 SYSTEM DASHBOARD":
        
        # TOP KPI ROWS (The 4 Cards from design)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown('<div class="kpi-card"><h3>Avg Portfolio Score</h3><p>710</p><p class="kpi-card-sub">FAIR & STABLE</p></div>', unsafe_allow_html=True)
        with kpi2:
            st.markdown('<div class="kpi-card"><h3>Predicted Default</h3><p>19.8%</p><p class="kpi-card-sub">BELOW THRESHOLD</p></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown('<div class="kpi-card"><h3>Active Models</h3><p>01</p><p class="kpi-card-sub">RANDOM FOREST</p></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown('<div class="kpi-card"><h3>System Accuracy</h3><p>80.17%</p><p class="kpi-card-sub">OPTIMIZED</p></div>', unsafe_allow_html=True)

        st.write("") # Spacer

        # MAIN CONTENT ROWS (Apply for Loan, Analysis Cards from design)
        col_form, col_anal = st.columns([1, 1.2])
        
        # 👤 Apply for Loan Card
        with col_form:
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            st.markdown('<div class="content-container-header">👤 APPLICANT DATA ENTRY</div>', unsafe_allow_html=True)
            
            if pipeline is None:
                st.error("🚨 Pipeline artifact not found. Please run 'train_model.py' first.")
            else:
                applicant_name = st.text_input("Applicant Name", placeholder="E.g., John Doe")
                
                # Input fields matching the UI design
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000.0, value=15000.0, step=500.0)
                    int_rate = st.number_input("Interest Rate (%)", value=10.5)
                with col_f2:
                    annual_inc = st.number_input("Annual Income ($)", value=75000.0)
                    term = st.selectbox("Loan Term (Months)", [36, 60])
                
                # Hidden features required by model but not needed in clean UI
                dti_val = 15.0 
                open_acc_val = 10 
                
                st.write("")
                submit = st.button("EXECUTE RISK ANALYSIS") 
                
            st.markdown('</div>', unsafe_allow_html=True)

        # 📈 Analysis / Results Card
        with col_anal:
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            st.markdown('<div class="content-container-header">📈 AI INFERENCE RESULTS</div>', unsafe_allow_html=True)
            
            if pipeline is not None and 'submit' in locals() and submit:
                with st.spinner("AI Engine executing risk analysis..."):
                    time.sleep(1) # For UI processing effect
                    
                    # Calculate dummy installment based on inputs
                    installment_val = (loan_amnt * (int_rate / 1200)) / (1 - (1 + int_rate / 1200)**(-term))
                    total_acc_val = open_acc_val * 2
                    
                    df_input = pd.DataFrame([[loan_amnt, term, int_rate, installment_val, annual_inc, dti_val, open_acc_val, total_acc_val]],
                                            columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                    
                    prediction = pipeline.predict(df_input)
                    probability = pipeline.predict_proba(df_input)[0][1] # Approval Probability
                    
                    st.write("")
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
                    
                    # Complete Plotly Chart block
                    st.write("---")
                    st.markdown("**Probability Benchmark**")
                    df_plot = pd.DataFrame({
                        "Metric": ["Applicant Confidence", "Approval Threshold"],
                        "Score": [probability*100, 60.0] # Assuming 60 is a threshold
                    })
                    
                    fig = px.bar(df_plot, x="Score", y="Metric", orientation='h', 
                                 color="Metric", color_discrete_sequence=['#00f2fe', '#374151'])
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb'),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=150,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Awaiting input data. Click 'Execute Risk Analysis' to generate AI insights.")
                
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # OTHER MODULES
    # ==========================================
    elif app_mode == "📂 BULK PROCESSING":
        st.markdown('<div class="content-container"><div class="content-container-header">📂 HIGH-VOLUME BATCH PROCESSING</div>', unsafe_allow_html=True)
        st.write("Upload a CSV file containing multiple customer records.")
        uploaded_file = st.file_uploader("Upload Batch CSV", type="csv")
        st.markdown('</div>', unsafe_allow_html=True)

    elif app_mode == "🗺️ CREDIT ROADMAP":
        st.markdown('<div class="content-container"><div class="content-container-header">🗺️ PERSONALISED CREDIT ROADMAP</div>', unsafe_allow_html=True)
        st.write("Generate a custom financial improvement plan for rejected clients.")
        target_score = st.slider("Target Credit Score", 300, 850, 750)
        st.button("GENERATE ROADMAP")
        st.markdown('</div>', unsafe_allow_html=True)

    elif app_mode == "🔒 ADMIN GATEWAY":
        st.markdown('<div class="content-container"><div class="content-container-header">🔒 RESTRICTED ACCESS</div>', unsafe_allow_html=True)
        user = st.text_input("Admin ID (Username)")
        pwd = st.text_input("Security Key (Password)", type="password")
        if st.button("AUTHENTICATE SESSION"):
            if authenticate(user, pwd):
                st.success("✅ Access Granted. System metrics online.")
            else:
                st.error("🚫 Authentication Failed.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
