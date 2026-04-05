Live Demo: [Click here to view the app] https://credit-eligibility-check.streamlit.app/

🚀 Loan Approval Prediction using Machine Learning
This project aims to predict the likelihood of a loan being approved or paid back based on various customer attributes such as credit score, income, debt-to-income ratio, and more. It utilizes a dataset of 20,000 records to train a robust Machine Learning model.

📊 Project Overview
Predicting loan defaults is a critical task for financial institutions. This project follows a structured data science workflow, from data cleaning and exploration to model deployment.

Key Features:
Comprehensive EDA: Detailed analysis of features influencing loan repayment.
Data Preprocessing: Handling categorical variables and feature scaling.
Machine Learning: Implementation of a Random Forest Classifier for high accuracy.
Model Persistence: The trained model is saved as a .sav file for production use.
Web App: A Streamlit-based web interface for real-time predictions.

🏗️ Project Structure
The repository is organized into the following sections:
Importing Dependencies: Loading necessary libraries (Pandas, Scikit-learn, etc.).
Data Cleaning: Handling missing values and duplicates.
Data Visualization: Graphical representation of data distributions.
EDA: Identifying correlations and key insights.
Data Modeling: Encoding and preparing features for the model.
Train-Test Split: Dividing data into 80% training and 20% testing sets.
Model Training: Training the Random Forest model.
Evaluation: Measuring performance using Accuracy Score and Confusion Matrix.
Model Saving: Exporting the model to loan_model.sav.

🛠️ Tech Stack
Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Joblib
Deployment: Streamlit (Web App)

🚀 How to Run
https://github.com/harimurugann/loan_approval_prediction

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app: https://credit-eligibility-check.streamlit.app/


📈 Results
The model achieves a high accuracy in predicting loan repayment, helping to automate the credit risk assessment process efficiently.
