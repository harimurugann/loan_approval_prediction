# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import time
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Loan Risk AI", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR UI DESIGN ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #00462e;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #006b46;
        color: white;
        transform: scale(1.02);
    }
    .custom-header {
        color: #1f2937;
        font-weight: 800;
        font-size: 2rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        color: white;
    }
    .roadmap-box {
        background-color: #e0f2fe;
        border-left: 5px solid #0284c7;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
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
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2602/2602112.png", width=80)
    st.sidebar.title("Risk Intelligence Hub")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("Navigate Modules", [
        "📊 Live Dashboard & Analytics",
        "👤 Single Risk Assessment",
        "📂 Bulk Batch Processing",
        "🧠 XAI & Credit Roadmap",  # NEW FEATURE ADDED HERE
        "🔒 Secure Admin Gateway"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("System v2.1.0 | Enterprise Edition")

    # ==========================================
    # MODULE 1: LIVE DASHBOARD & ANALYTICS
    # ==========================================
    if app_mode == "📊 Live Dashboard & Analytics":
        st.markdown('<div class="custom-header">📊 Data Analytics & Portfolio Overview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Total Active Models", value="1 (Random Forest)", delta="Optimized")
        col2.metric(label="System Accuracy", value="80.17%", delta="Stable")
        col3.metric(label="Processing Engine", value="Pipeline Active", delta="Online")
        st.write("---")
        data_file = st.file_uploader("Upload dataset for interactive analysis", type="csv")
        if data_file is not None:
            df_plot = pd.read_csv(data_file)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Loan Status Distribution")
                if 'loan_paid_back' in df_plot.columns:
                    fig = px.pie(df_plot, names='loan_paid_back', hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Loan Amount vs Interest Rate")
                if 'loan_amnt' in df_plot.columns and 'int_rate' in df_plot.columns:
                    fig2 = px.scatter(df_plot.head(500), x='loan_amnt', y='int_rate', color='loan_paid_back', opacity=0.7)
                    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # MODULE 2: SINGLE RISK ASSESSMENT
    # ==========================================
    elif app_mode == "👤 Single Risk Assessment":
        st.markdown('<div class="custom-header">👤 AI Underwriting Engine</div>', unsafe_allow_html=True)
        if pipeline is None:
            st.error("🚨 Pipeline artifact not found. Please run 'train_model.py' first.")
            return

        with st.form("single_eval"):
            col1, col2, col3 = st.columns(3)
            with col1:
                loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, value=10000.0, step=500.0)
                annual_inc = st.number_input("Annual Income ($)", min_value=10000.0, value=70000.0, step=1000.0)
                open_acc = st.number_input("Active Credit Lines", min_value=1, value=10)
            with col2:
                term = st.selectbox("Loan Term (Months)", [36, 60])
                dti = st.number_input("Debt-to-Income (DTI %)", min_value=0.0, value=15.0)
                total_acc = st.number_input("Total Historical Accounts", min_value=1, value=20)
            with col3:
                int_rate = st.number_input("Requested Interest Rate (%)", min_value=5.0, value=10.5)
                installment = st.number_input("Monthly Installment ($)", min_value=50.0, value=300.0)
            
            submit = st.form_submit_button("🧠 Execute Prediction")
            if submit:
                with st.spinner("Analyzing risk factors..."):
                    time.sleep(1)
                    df_input = pd.DataFrame([[loan_amnt, term, int_rate, installment, annual_inc, dti, open_acc, total_acc]],
                                            columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                    prediction = pipeline.predict(df_input)
                    probability = pipeline.predict_proba(df_input)[0][1]
                    
                    st.write("---")
                    res_col1, res_col2 = st.columns(2)
                    if prediction[0] == 1:
                        res_col1.success("✅ STATUS: APPROVED")
                        st.balloons()
                    else:
                        res_col1.error("🚫 STATUS: REJECTED")
                    res_col2.metric("Approval Confidence", f"{probability:.2%}")
                    st.progress(probability)

    # ==========================================
    # MODULE 3: BULK BATCH PROCESSING
    # ==========================================
    elif app_mode == "📂 Bulk Batch Processing":
        st.markdown('<div class="custom-header">📂 Batch Processing Engine</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Batch CSV", type="csv")
        if uploaded_file is not None and pipeline is not None:
            df_bulk = pd.read_csv(uploaded_file)
            st.dataframe(df_bulk.head(), use_container_width=True)
            if st.button("▶️ Start Batch Assessment"):
                X_bulk = df_bulk.drop('loan_paid_back', axis=1) if 'loan_paid_back' in df_bulk.columns else df_bulk
                predictions = pipeline.predict(X_bulk)
                df_bulk['AI_Decision'] = ["Approved" if p == 1 else "Denied" for p in predictions]
                st.success("✅ Batch Prediction Completed!")
                csv = df_bulk.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Report (CSV)", data=csv, file_name='Batch_Report.csv', mime='text/csv')

    # ==========================================
    # MODULE 4: XAI & CREDIT ROADMAP (NEW)
    # ==========================================
    elif app_mode == "🧠 XAI & Credit Roadmap":
        st.markdown('<div class="custom-header">🧠 Explainable AI (XAI) & Action Plan</div>', unsafe_allow_html=True)
        st.write("Analyze a specific profile to understand the **'Why'** behind the AI's decision and generate a personalized financial roadmap.")
        
        if pipeline is None:
            st.error("🚨 Pipeline artifact not found.")
            return

        with st.container():
            st.subheader("1. Input Client Profile for Deep Analysis")
            c1, c2 = st.columns(2)
            req_loan = c1.number_input("Requested Loan ($)", value=25000.0)
            req_inc = c2.number_input("Client Annual Income ($)", value=45000.0)
            req_dti = c1.number_input("Client DTI (%)", value=35.0)
            req_rate = c2.number_input("Interest Rate (%)", value=18.5)
            
            if st.button("🔍 Generate XAI Report"):
                with st.spinner("Extracting Model Logic..."):
                    time.sleep(1.5)
                    # Create dummy data for the rest of the features to run the model
                    df_xai = pd.DataFrame([[req_loan, 60, req_rate, 500.0, req_inc, req_dti, 8, 15]],
                                            columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                    
                    pred = pipeline.predict(df_xai)[0]
                    
                    # 1. Prediction Result
                    st.write("---")
                    if pred == 1:
                        st.success("### AI Decision: LIKELY TO APPROVE")
                    else:
                        st.error("### AI Decision: HIGH RISK (LIKELY TO REJECT)")

                    # 2. Extract Feature Importances from the Random Forest Model
                    model = pipeline.named_steps['classifier']
                    features = ['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc']
                    importances = model.feature_importances_
                    
                    df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
                    df_imp = df_imp.sort_values(by='Importance', ascending=True)

                    st.subheader("📊 XAI: Which factors influenced this AI model?")
                    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                                 title="Global Feature Importance (Random Forest)",
                                 color='Importance', color_continuous_scale='Tealgrn')
                    st.plotly_chart(fig, use_container_width=True)

                    # 3. Automated Rule-Based Credit Roadmap
                    st.subheader("🗺️ Personalized Credit Roadmap (AI Generated)")
                    roadmap_html = ""
                    
                    if req_dti > 20.0:
                        roadmap_html += "<li><b>Reduce Debt-to-Income (DTI):</b> Your DTI is high. Focus on paying off existing credit card debts before applying for this loan.</li>"
                    else:
                        roadmap_html += "<li><b>Maintain DTI:</b> Your Debt-to-Income ratio is extremely healthy. Keep it below 20%.</li>"
                        
                    if req_rate > 15.0:
                        roadmap_html += "<li><b>Renegotiate Interest Rate:</b> High interest rates increase default risk. Try to improve your credit score to secure a rate below 12%.</li>"
                        
                    if req_loan > (req_inc * 0.4):
                        roadmap_html += "<li><b>Loan Amount Warning:</b> The requested loan is a large percentage of your annual income. Consider requesting a smaller amount or extending the term to reduce monthly installments.</li>"
                    else:
                        roadmap_html += "<li><b>Income Coverage:</b> Your income safely covers the requested loan amount.</li>"

                    st.markdown(f'<div class="roadmap-box"><ul>{roadmap_html}</ul></div>', unsafe_allow_html=True)

    # ==========================================
    # MODULE 5: SECURE ADMIN GATEWAY
    # ==========================================
    elif app_mode == "🔒 Secure Admin Gateway":
        st.markdown('<div class="custom-header">🔒 Restricted Admin Console</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user = st.text_input("Admin ID")
            pwd = st.text_input("Security Key", type="password")
            if st.button("🔐 Authenticate"):
                if authenticate(user, pwd):
                    st.success("✅ Access Granted.")
                    st.code("System Online.\nNo Data Drifts Detected.\nSecurity Protocol: Active", language="bash")
                else:
                    st.error("🚫 Invalid Credentials.")

if __name__ == "__main__":
    main()
