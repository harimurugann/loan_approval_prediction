"""
=============================================================================
  LOAN APPROVAL INTELLIGENCE SYSTEM — Training Pipeline
  Author  : Senior ML Engineer
  Dataset : loan_dataset_20000.csv  (20,000 records, 22 columns)
  Models  : RandomForestClassifier (champion) + XGBoost (challenger)
  Outputs : loan_rf_model.sav, loan_xgb_model.sav, loan_pipeline.sav
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 ─ Importing the Dependencies
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                         # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed – challenger model will be skipped.")

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─ Output folders ─────────────────────────────────────────────────────────────
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=" * 65)
print("  LOAN APPROVAL INTELLIGENCE SYSTEM  —  Training Pipeline")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 ─ Data Cleaning (Nulls · Duplicates · Outliers)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Data Cleaning …")

df = pd.read_csv("loan_dataset_20000.csv")
print(f"  Raw shape : {df.shape}")

# ── Null handling ──────────────────────────────────────────────────────────
null_counts = df.isnull().sum()
print(f"  Total nulls: {null_counts.sum()}")
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ── Duplicate removal ──────────────────────────────────────────────────────
dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"  Duplicates dropped: {dupes}  → new shape: {df.shape}")

# ── Outlier capping with IQR (numeric columns only, skip target) ───────────
NUM_COLS = [
    "annual_income", "monthly_income", "loan_amount", "installment",
    "total_credit_limit", "current_balance", "credit_score"
]
for col in NUM_COLS:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR     = Q3 - Q1
    lo, hi  = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    before  = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)
    print(f"  Outlier-capped  {col:30s}: {before} values clipped")

print(f"  Clean shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 ─ Data Visualization
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Data Visualization …")

# Distribution of key numeric features
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")
plot_cols = ["age", "credit_score", "annual_income", "loan_amount",
             "debt_to_income_ratio", "interest_rate", "installment", "num_of_delinquencies"]
for ax, col in zip(axes.flatten(), plot_cols):
    ax.hist(df[col], bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig("reports/feature_distributions.png", dpi=120)
plt.close()

# Loan outcome by categorical features
cat_cols = ["gender", "education_level", "employment_status", "loan_purpose", "marital_status"]
fig, axes = plt.subplots(1, len(cat_cols), figsize=(22, 5))
fig.suptitle("Loan Paid Back Rate by Category", fontsize=14, fontweight="bold")
for ax, col in zip(axes, cat_cols):
    rates = df.groupby(col)["loan_paid_back"].mean().sort_values()
    rates.plot(kind="barh", ax=ax, color="#4CAF50", edgecolor="white")
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("Approval Rate")
plt.tight_layout()
plt.savefig("reports/categorical_approval_rates.png", dpi=120)
plt.close()

print("  Saved → reports/feature_distributions.png")
print("  Saved → reports/categorical_approval_rates.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 ─ Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4] EDA …")

# Correlation heatmap
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()
plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.4,
            annot_kws={"size": 8})
plt.title("Correlation Matrix", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png", dpi=120)
plt.close()

# Feature impact on target (point-biserial approximation via correlation)
target_corr = numeric_df.corr()["loan_paid_back"].drop("loan_paid_back").abs().sort_values(ascending=True)
plt.figure(figsize=(9, 6))
target_corr.plot(kind="barh", color="#FF9800")
plt.title("Feature Correlation with Loan Approval", fontsize=13, fontweight="bold")
plt.xlabel("|Pearson r|")
plt.tight_layout()
plt.savefig("reports/feature_impact.png", dpi=120)
plt.close()

print("  Saved → reports/correlation_heatmap.png")
print("  Saved → reports/feature_impact.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 ─ Data Modeling — Scikit-learn Pipeline Design
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Pipeline Design …")

CATEGORICAL_FEATURES = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]
NUMERICAL_FEATURES = [
    "age", "annual_income", "monthly_income", "debt_to_income_ratio",
    "credit_score", "loan_amount", "interest_rate", "loan_term",
    "installment", "num_of_open_accounts", "total_credit_limit",
    "current_balance", "delinquency_history", "public_records",
    "num_of_delinquencies"
]
TARGET = "loan_paid_back"

# Numerical sub-pipeline: median imputation + standardisation
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Categorical sub-pipeline: most-frequent imputation + OHE
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Column transformer combining both pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, NUMERICAL_FEATURES),
    ("cat", cat_pipeline, CATEGORICAL_FEATURES)
])

print("  Preprocessor built with ColumnTransformer.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 ─ Split Features and Target
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 6] Splitting Features / Target …")
X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]
print(f"  X shape: {X.shape}  |  y distribution: {dict(y.value_counts())}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 ─ Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7] Train / Test Split (80/20, stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 ─ Model Training
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 8] Model Training …")

# ── Champion: Random Forest ────────────────────────────────────────────────
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ))
])
print("  Training RandomForestClassifier …")
rf_pipeline.fit(X_train, y_train)
print("  RandomForest trained ✓")

# ── Challenger: XGBoost ────────────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        ))
    ])
    print("  Training XGBClassifier …")
    xgb_pipeline.fit(X_train, y_train)
    print("  XGBoost trained ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 ─ Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 9] Model Evaluation …")

def evaluate_model(name, pipeline, X_t, y_t):
    """Compute and print full classification metrics."""
    y_pred = pipeline.predict(X_t)
    y_prob = pipeline.predict_proba(X_t)[:, 1]
    acc   = accuracy_score(y_t, y_pred)
    prec  = precision_score(y_t, y_pred)
    rec   = recall_score(y_t, y_pred)
    f1    = f1_score(y_t, y_pred)
    auc   = roc_auc_score(y_t, y_prob)
    print(f"\n  ── {name} ──")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_t, y_pred, target_names=["Rejected", "Approved"]))
    return {"name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc,
            "y_pred": y_pred, "y_prob": y_prob}

rf_metrics  = evaluate_model("RandomForest (Champion)", rf_pipeline, X_test, y_test)
if XGBOOST_AVAILABLE:
    xgb_metrics = evaluate_model("XGBoost (Challenger)", xgb_pipeline, X_test, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 ─ Communication & Visualization of Results
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 10] Visualising Results …")

def plot_confusion(name, y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()

plot_confusion("RandomForest", y_test, rf_metrics["y_pred"],
               "reports/cm_random_forest.png")

if XGBOOST_AVAILABLE:
    plot_confusion("XGBoost", y_test, xgb_metrics["y_pred"],
                   "reports/cm_xgboost.png")

# ROC Curve comparison
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, rf_metrics["y_prob"])
plt.plot(fpr, tpr, label=f"RandomForest AUC={rf_metrics['roc_auc']:.3f}", lw=2)
if XGBOOST_AVAILABLE:
    fpr2, tpr2, _ = roc_curve(y_test, xgb_metrics["y_prob"])
    plt.plot(fpr2, tpr2, label=f"XGBoost AUC={xgb_metrics['roc_auc']:.3f}", lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Champion vs Challenger", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("reports/roc_curves.png", dpi=120)
plt.close()

# Feature importance (RandomForest)
rf_clf   = rf_pipeline.named_steps["classifier"]
ohe_cols = rf_pipeline.named_steps["preprocessor"] \
              .named_transformers_["cat"] \
              .named_steps["encoder"] \
              .get_feature_names_out(CATEGORICAL_FEATURES)
all_feat = NUMERICAL_FEATURES + list(ohe_cols)
importances = pd.Series(rf_clf.feature_importances_, index=all_feat) \
                .sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
importances[::-1].plot(kind="barh", color="#2196F3", edgecolor="white")
plt.title("Top-20 Feature Importances (RandomForest)", fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importances.png", dpi=120)
plt.close()

# Save top importances for Streamlit to consume
importances.to_csv("reports/feature_importances.csv", header=True)

print("  Saved → reports/cm_random_forest.png")
print("  Saved → reports/roc_curves.png")
print("  Saved → reports/feature_importances.png")
print("  Saved → reports/feature_importances.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 ─ Model Saving (.sav via joblib)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 11] Saving Models (.sav) …")
joblib.dump(rf_pipeline, "models/loan_rf_model.sav")
print("  Saved → models/loan_rf_model.sav")

if XGBOOST_AVAILABLE:
    joblib.dump(xgb_pipeline, "models/loan_xgb_model.sav")
    print("  Saved → models/loan_xgb_model.sav")

# Save metrics for Streamlit benchmarking tab
metrics_dict = {
    "RandomForest": {k: v for k, v in rf_metrics.items()
                     if k not in ("y_pred", "y_prob")}
}
if XGBOOST_AVAILABLE:
    metrics_dict["XGBoost"] = {k: v for k, v in xgb_metrics.items()
                                if k not in ("y_pred", "y_prob")}
joblib.dump(metrics_dict, "models/benchmark_metrics.sav")
print("  Saved → models/benchmark_metrics.sav")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 ─ Automated Pipeline Implementation (Preprocessing + Prediction)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 12] Automated Pipeline Verification …")

def predict_batch(pipeline, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run end-to-end prediction on a raw input DataFrame.
    Returns the original frame with appended prediction columns.
    """
    probs   = pipeline.predict_proba(input_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    labels  = (probs >= 0.50).astype(int)
    result  = input_df.copy()
    result["approval_probability"] = probs.round(4)
    result["prediction"]           = labels
    result["risk_label"]           = pd.cut(probs,
        bins=[-0.01, 0.40, 0.65, 1.01],
        labels=["High Risk", "Moderate Risk", "Low Risk"])
    return result

sample = X_test.head(5).copy()
sample["loan_paid_back"] = y_test.head(5).values
result_df = predict_batch(rf_pipeline, sample)
print("  Batch prediction sample:")
print(result_df[["approval_probability", "prediction", "risk_label"]].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 ─ Pipeline Saving (.sav)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 13] Saving Full Inference Pipeline …")

inference_bundle = {
    "pipeline":             rf_pipeline,
    "numerical_features":   NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "target":               TARGET,
    "predict_batch_fn":     predict_batch,
    "label_map":            {0: "Rejected", 1: "Approved"},
}
joblib.dump(inference_bundle, "models/loan_pipeline.sav")
print("  Saved → models/loan_pipeline.sav")

# Save column metadata for Streamlit
meta = {
    "NUMERICAL_FEATURES":   NUMERICAL_FEATURES,
    "CATEGORICAL_FEATURES": CATEGORICAL_FEATURES,
    "TARGET":               TARGET,
    "CATEGORICAL_VALUES": {
        "gender":           df["gender"].unique().tolist(),
        "marital_status":   df["marital_status"].unique().tolist(),
        "education_level":  df["education_level"].unique().tolist(),
        "employment_status":df["employment_status"].unique().tolist(),
        "loan_purpose":     df["loan_purpose"].unique().tolist(),
        "grade_subgrade":   df["grade_subgrade"].unique().tolist(),
    },
    "NUMERIC_RANGES": {col: {"min": float(df[col].min()), "max": float(df[col].max()),
                              "mean": float(df[col].mean())}
                       for col in NUMERICAL_FEATURES}
}
joblib.dump(meta, "models/column_meta.sav")
print("  Saved → models/column_meta.sav")

# Anomaly detector (IsolationForest on numeric features after preprocessing)
X_num = num_pipeline.fit_transform(df[NUMERICAL_FEATURES])
iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso_forest.fit(X_num)
joblib.dump(iso_forest,   "models/anomaly_detector.sav")
joblib.dump(num_pipeline, "models/num_scaler.sav")
print("  Saved → models/anomaly_detector.sav")
print("  Saved → models/num_scaler.sav")

print("\n" + "=" * 65)
print("  Training Pipeline Complete — All artefacts saved to ./models/")
print("=" * 65)
