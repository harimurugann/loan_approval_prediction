"""Project-wide constants: paths, column names, and modelling settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "loan_dataset_20000.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "loan_model.joblib"
REPORT_PATH = MODEL_DIR / "training_report.json"

TARGET = "loan_paid_back"
FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "open_acc",
    "total_acc",
]

FEATURE_LABELS = {
    "loan_amnt": "Loan amount",
    "term": "Term",
    "int_rate": "Interest rate (%)",
    "installment": "Monthly installment",
    "annual_inc": "Annual income",
    "dti": "Debt-to-income ratio (%)",
    "open_acc": "Open accounts",
    "total_acc": "Total accounts",
}

FEATURE_DESCRIPTIONS = {
    "loan_amnt": "Amount of the loan requested",
    "term": "Repayment period in months",
    "int_rate": "Annual interest rate on the loan",
    "installment": "Monthly payment owed on the loan",
    "annual_inc": "Applicant's yearly income",
    "dti": "Monthly debt payments divided by monthly income",
    "open_acc": "Credit accounts currently open",
    "total_acc": "Credit accounts ever opened",
}

VALID_TERMS = (36, 60)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5
MIN_USEFUL_AUC = 0.60
