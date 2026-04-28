# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import bcrypt

# Directory Handling
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'full_pipeline.sav')

# Page Configuration
st.set_page_config(page_title="Enterprise Loan Risk Intelligence", layout="wide")

@st.cache_resource
def load_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = load_pipeline()

def authenticate(username, password):
    """Secure Admin Gateway Authentication"""
    hashed = b'$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjIQqiRQYq' # Hash for 'admin123'
    return username == "admin" and bcrypt.checkpw(password.encode('utf-8'), hashed)

def main():
    st.sidebar.title("Enterprise Control Panel")
    app_mode = st.sidebar.selectbox("Intelligence Modules", [
        "1. Single Assessment Hub",
        "2. Bulk Assessment Hub",
        "3. Live Dashboard & Analytics",
        "4. Secure Admin Gateway",
        "5. Other Modules (Placeholders)"
    ])

    if app_mode == "1. Single Assessment Hub":
        st.header("Single Entity Risk Assessment")
        
        if pipeline is None:
            st.error("Pipeline artifact not found. Please run 'train_model.py' first.")
            return

        with st.form("single_eval"):
            col1, col2 = st.columns(2)
            loan_amnt = col1.number_input("Loan Amount", value=10000.0)
            term = col1.selectbox("Term (Months)", [36, 60])
            int_rate = col2.number_input("Interest Rate (%)", value=10.5)
            annual_inc = col2.number_input("Annual Income", value=70000.0)
            installment = col1.number_input("Installment", value=300.0)
            dti = col2.number_input("DTI", value=15.0)
            open_acc = col1.number_input("Open Accounts", value=10)
            total_acc = col2.number_input("Total Accounts", value=20)
            
            submit = st.form_submit_button("Execute ML Prediction")
            
            if submit:
                df_input = pd.DataFrame([[loan_amnt, term, int_rate, installment, annual_inc, dti, open_acc, total_acc]],
                                        columns=['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc'])
                
                prediction = pipeline.predict(df_input)
                probability = pipeline.predict_proba(df_input)[0][1]
                
                if prediction[0] == 1:
                    st.success(f"Assessment Status: APPROVED (Probability Score: {probability:.2%})")
                else:
                    st.error(f"Assessment Status: DENIED / HIGH RISK (Probability Score: {probability:.2%})")

    elif app_mode == "2. Bulk Assessment Hub":
        st.header("Bulk Assessment (Upload CSV)")
        st.write("Upload a CSV file with customer data to get batch predictions.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None and pipeline is not None:
            df_bulk = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(df_bulk.head())
            
            if st.button("Run Bulk Prediction"):
                try:
                    # Drop target column if it exists in the uploaded file
                    if 'loan_paid_back' in df_bulk.columns:
                        X_bulk = df_bulk.drop('loan_paid_back', axis=1)
                    else:
                        X_bulk = df_bulk
                        
                    predictions = pipeline.predict(X_bulk)
                    df_bulk['Predicted_Status'] = ["Approved" if p == 1 else "Denied" for p in predictions]
                    
                    st.success("Bulk Prediction Completed!")
                    st.dataframe(df_bulk[['loan_amnt', 'annual_inc', 'Predicted_Status']].head(10))
                    
                    # Convert to CSV for download
                    csv = df_bulk.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='bulk_predictions_result.csv',
                        mime='text/csv',
                    )
                except Exception as e:
                    st.error(f"Error in prediction. Make sure column names match the training data. Error: {e}")

    elif app_mode == "3. Live Dashboard & Analytics":
        st.header("Data Analytics")
        st.info("Here you can visualize the data. Try uploading the 'loan_dataset_20000.csv' file.")
        data_file = st.file_uploader("Upload dataset for analysis", type="csv", key="analytics")
        if data_file is not None:
            df_plot = pd.read_csv(data_file)
            st.write("Summary Statistics")
            st.write(df_plot.describe())
            
            if 'loan_amnt' in df_plot.columns:
                st.subheader("Loan Amount Distribution")
                st.bar_chart(df_plot['loan_amnt'].head(50)) # Showing first 50 for quick rendering

    elif app_mode == "4. Secure Admin Gateway":
        st.header("Restricted Access")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Authenticate"):
            if authenticate(user, pwd):
                st.success("Authorized: Level 4 Security Clearance Granted.")
            else:
                st.error("Authentication Failed. (Hint: use admin / admin123)")
                
    elif app_mode == "5. Other Modules (Placeholders)":
        st.header("Under Construction")
        st.warning("Module interface is active. Connect underlying datastore in Production configuration.")
        st.write("Idhu madhiri matha modules-a (PDF generation, Geospatial mapping) unga thevaikku yetpa code add pannikkalam.")

if __name__ == "__main__":
    main()
