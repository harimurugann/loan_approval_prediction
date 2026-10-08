Loan Approval Prediction

A Streamlit application that trains a scikit-learn classifier on 20,000 historical loan records and estimates the probability that a loan is paid back. It includes an input form with validation, an analytics dashboard, and a model performance page that reports metrics computed from the real held-out test set.

Important finding. On this dataset every feature has a near-zero correlation with loan_paid_back. Three model families all reach a cross-validated ROC-AUC of about 0.50 (chance level), and the final model predicts "paid back" for every applicant, matching the 80.2% majority-class baseline. The app reports this openly. See Limitations.

Features
Home: key statistics, model overview, feature table
Loan Prediction: validated form, model-estimated probability, input summary
Analytics: distributions, repayment rate by loan amount / interest rate / term, correlation matrix, data quality
Model Performance: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, permutation importance, classification report
About: methodology, limitations, responsible-use note
Dataset

data/loan_dataset_20000.csv: 20,000 rows, 9 columns.

Column	Type	Notes
loan_amnt, int_rate, installment, annual_inc, dti	float	annual_inc has 500 missing, dti has 200 missing
term	int	36 or 60 months
open_acc, total_acc	int	3,619 rows have open_acc > total_acc
loan_paid_back	int (target)	1 = paid back (80.2%), 0 = not paid back (19.8%)

No duplicate rows and no IQR outliers were found.

ML workflow
Load and validate the CSV (required columns, numeric types, binary target).
Stratified 80/20 train/test split before any fitting.
Pipelines: median imputation (+ standard scaling for logistic regression).
Compare Logistic Regression, Random Forest, and Hist Gradient Boosting using 5-fold cross-validated ROC-AUC on the training set only.
Select with the one-standard-error rule (simplest model within one standard deviation of the best score).
Evaluate once on the test set; save the full pipeline (joblib) and a JSON training report.

Leakage controls: imputation and scaling are inside the pipeline, so they are fitted on training data only, and the same fitted pipeline transforms user input at prediction time. The test set is never used for selection.

Project structure
text
loan-approval-prediction/
├── app.py                      Streamlit entry point and navigation
├── train_model.py              Reproducible training script
├── requirements.txt
├── README.md
├── .streamlit/config.toml      Theme
├── data/loan_dataset_20000.csv
├── models/                     loan_model.joblib + training_report.json (generated)
├── app_pages/                  home, prediction, analytics, model_performance, about
├── components/                 theme (CSS), ui (cards), charts (Plotly)
└── utils/                      config, data_loader, model_utils, validation,
                                prediction, analysis, resources (cached loaders)
Installation (Windows)
bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py

macOS / Linux: use source venv/bin/activate instead of venv\Scripts\activate.

python train_model.py is optional. If the model files are missing, corrupt, or built with a different scikit-learn version, the app trains and saves a new model automatically on first run (a few seconds).

Python 3.10 or newer is recommended.

Example usage

Open the Loan Prediction page, keep the default values (dataset medians) or edit them, and select Predict repayment. Out-of-range or impossible inputs (negative amounts, a term other than 36/60, open accounts above total accounts) are rejected with a specific message.

Limitations
The features carry almost no signal: test ROC-AUC is about 0.51, and the model never predicts "not paid back". Estimates sit near the dataset's overall repayment rate.
The data is not internally consistent (open_acc > total_acc in 18% of rows; installments unrelated to amount, rate, and term) and may be synthetic.
Only eight fields are used. No credit history, employment, or macroeconomic data.
The target is loan repayment in this dataset, not a bank's approval decision.
Inputs outside the training ranges are rejected rather than extrapolated.

Responsible use
This project is for demonstration, education, and portfolio purposes. Predictions reflect historical patterns in one dataset, are not guaranteed, and may embed bias from the training data. Real lending decisions require additional financial, regulatory, and human review. Do not use this application to make decisions about real people.
