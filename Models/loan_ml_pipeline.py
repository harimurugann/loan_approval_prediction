"""
=============================================================================
  Enterprise Loan Risk Intelligence System — ML Pipeline
  13-Module Production-Grade Implementation
=============================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# MODULE 1 — ENVIRONMENT & DEPENDENCY ORCHESTRATION
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)

# Absolute path anchors
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
MODEL_DIR  = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH         = os.path.join(DATA_DIR,  "loan_dataset_20000.csv")
MODEL_PATH        = os.path.join(MODEL_DIR, "champion_model.sav")
PIPELINE_PATH     = os.path.join(MODEL_DIR, "full_pipeline.sav")

print("=" * 60)
print("  MODULE 1 — Environment orchestration complete")
print("=" * 60)

# ---------------------------------------------------------------------------
# MODULE 2 — DATA INTEGRITY & CLEANING
# ---------------------------------------------------------------------------
print("\n[MODULE 2] Loading and sanitising dataset …")

df_raw = pd.read_csv(DATA_PATH)
print(f"  Raw shape          : {df_raw.shape}")
print(f"  Null counts:\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0]}")
print(f"  Duplicates         : {df_raw.duplicated().sum()}")

# Drop true duplicates
df = df_raw.drop_duplicates().copy()

# Drop identifier column — no predictive value
if "customer_id" in df.columns:
    df.drop(columns=["customer_id"], inplace=True)

# Sanitise: clip obvious outliers to valid domain
df["credit_score"]      = df["credit_score"].clip(300, 850)
df["debt_to_income"]    = df["debt_to_income"].clip(0, 100)
df["interest_rate"]     = df["interest_rate"].clip(1, 40)
df["annual_income"]     = df["annual_income"].clip(0, None)
df["loan_amount"]       = df["loan_amount"].clip(0, None)
df["employment_years"]  = df["employment_years"].clip(0, 50)
df["num_credit_lines"]  = df["num_credit_lines"].clip(0, None)
df["delinquencies_2yrs"]= df["delinquencies_2yrs"].clip(0, None)

# Derived risk features (domain-informed feature engineering)
df["loan_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1e-6)
df["payment_burden"]    = df["debt_to_income"] * df["loan_amount"] / (df["annual_income"] + 1e-6)
df["risk_index"]        = (df["delinquencies_2yrs"] * 10 + (850 - df["credit_score"]) / 85)

print(f"  Clean shape        : {df.shape}")
print(f"  Target balance     : {df['loan_paid_back'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# MODULE 3 — STATISTICAL VISUALIZATION: Target Distribution
# ---------------------------------------------------------------------------
print("\n[MODULE 3] Generating target-distribution plots …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Loan Paid-Back Distribution", fontsize=15, fontweight="bold")

# Bar chart
counts = df["loan_paid_back"].value_counts()
axes[0].bar(["Not Paid (0)", "Paid (1)"], counts.values,
            color=["#e74c3c", "#2ecc71"], edgecolor="black", linewidth=0.8)
axes[0].set_title("Class Count")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10)

# Pie chart
axes[1].pie(counts.values, labels=["Not Paid", "Paid"],
            autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"],
            startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[1].set_title("Class Proportion")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: target_distribution.png")

# ---------------------------------------------------------------------------
# MODULE 4 — EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
print("\n[MODULE 4] EDA — correlation matrices and feature analysis …")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=axes[0], center=0, square=True,
            annot_kws={"size": 7})
axes[0].set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")

# Feature vs target box distributions
feature_order = (
    corr["loan_paid_back"].drop("loan_paid_back").abs()
    .sort_values(ascending=False).head(6).index.tolist()
)
axes[1].remove()
# Replace with a grid of violin plots
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle("Top 6 Features vs Loan Outcome", fontsize=13, fontweight="bold")
for ax, feat in zip(axes2.flatten(), feature_order):
    sub = df[[feat, "loan_paid_back"]].dropna()
    ax.violinplot(
        [sub.loc[sub["loan_paid_back"]==0, feat].values,
         sub.loc[sub["loan_paid_back"]==1, feat].values],
        positions=[0, 1], showmedians=True
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Paid", "Paid"])
    ax.set_title(feat, fontsize=9)
    ax.set_ylabel("Value", fontsize=8)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "eda_violin_plots.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "eda_correlation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: eda_correlation.png, eda_violin_plots.png")

# ---------------------------------------------------------------------------
# MODULE 5 — SCALABLE DATA MODELLING: ColumnTransformer Pipeline
# ---------------------------------------------------------------------------
print("\n[MODULE 5] Constructing preprocessing ColumnTransformer pipeline …")

NUMERIC_FEATURES = [
    "age", "annual_income", "loan_amount", "interest_rate",
    "employment_years", "credit_score", "debt_to_income",
    "num_credit_lines", "delinquencies_2yrs",
    "loan_income_ratio", "payment_burden", "risk_index"
]
CATEGORICAL_FEATURES = ["loan_purpose", "loan_grade", "home_ownership", "state"]
TARGET = "loan_paid_back"

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
], remainder="drop")

print(f"  Numeric  features : {len(NUMERIC_FEATURES)}")
print(f"  Categorical features : {len(CATEGORICAL_FEATURES)}")

# ---------------------------------------------------------------------------
# MODULE 6 — FEATURE-TARGET ARCHITECTURE
# ---------------------------------------------------------------------------
print("\n[MODULE 6] Decoupling features from target variable …")

all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = df[all_features]
y = df[TARGET]

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}  |  Positive rate: {y.mean():.3f}")

# ---------------------------------------------------------------------------
# MODULE 7 — STRATIFIED DATA PARTITIONING (80/20, seed=42)
# ---------------------------------------------------------------------------
print("\n[MODULE 7] Stratified train/test split 80/20 (seed=42) …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train set : {X_train.shape}  |  Test set : {X_test.shape}")
print(f"  Train +ve : {y_train.mean():.3f}  |  Test +ve : {y_test.mean():.3f}")

# ---------------------------------------------------------------------------
# MODULE 8 — MODEL TRAINING (Champion: Gradient Boosting)
# ---------------------------------------------------------------------------
print("\n[MODULE 8] Training champion model (GradientBoostingClassifier) …")

champion_clf = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.08,
    subsample=0.8, min_samples_split=20, random_state=42
)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   champion_clf)
])

full_pipeline.fit(X_train, y_train)
print("  Training complete.")

# Cross-validation sanity check
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"  5-fold CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# MODULE 9 — MODEL EVALUATION
# ---------------------------------------------------------------------------
print("\n[MODULE 9] Comprehensive model evaluation …")

y_pred  = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n  {'Metric':<20} {'Value':>10}")
print(f"  {'-'*32}")
print(f"  {'Accuracy':<20} {acc:>10.4f}")
print(f"  {'F1-Score':<20} {f1:>10.4f}")
print(f"  {'ROC-AUC':<20} {roc_auc:>10.4f}")
print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Not Paid','Paid'])}")

# ROC Curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {roc_auc:.4f}")
ax.plot([0,1],[0,1], "k--", lw=1)
ax.fill_between(fpr, tpr, alpha=0.1, color="#2980b9")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curve — Champion GBM")
ax.legend(loc="lower right", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: roc_curve.png")

# ---------------------------------------------------------------------------
# MODULE 10 — INSIGHTS COMMUNICATION
# ---------------------------------------------------------------------------
print("\n[MODULE 10] Generating confusion matrix and feature importance …")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Not Paid","Paid"], yticklabels=["Not Paid","Paid"])
axes[0].set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")

# Feature importance (from fitted GBM inside pipeline)
raw_importances = champion_clf.feature_importances_

# Reconstruct feature names after OHE
ohe_cats = (full_pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(CATEGORICAL_FEATURES))
feature_names = NUMERIC_FEATURES + list(ohe_cats)

imp_df = (pd.Series(raw_importances, index=feature_names)
          .sort_values(ascending=False)
          .head(15))

axes[1].barh(imp_df.index[::-1], imp_df.values[::-1], color="#1abc9c", edgecolor="black", linewidth=0.6)
axes[1].set(xlabel="Importance", title="Top 15 Feature Importances (GBM)")
axes[1].tick_params(axis="y", labelsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: confusion_feature_importance.png")

# ---------------------------------------------------------------------------
# MODULE 11 — ARTIFACT PERSISTENCE (.sav via joblib)
# ---------------------------------------------------------------------------
print("\n[MODULE 11] Persisting champion model as .sav artifact …")

joblib.dump(champion_clf, MODEL_PATH)
print(f"  Model saved  → {MODEL_PATH}")

# ---------------------------------------------------------------------------
# MODULE 12 — AUTOMATED INFERENCE PIPELINE
# ---------------------------------------------------------------------------
print("\n[MODULE 12] Wrapping prediction logic into single-call function …")

def predict_loan_risk(input_data: dict) -> dict:
    """
    Single-call inference entry point.

    Parameters
    ----------
    input_data : dict
        Keys matching the NUMERIC_FEATURES + CATEGORICAL_FEATURES lists.

    Returns
    -------
    dict with keys:
        prediction   – 1 (paid back) or 0 (not paid back)
        probability  – probability of repayment (class 1)
        risk_level   – 'LOW' | 'MEDIUM' | 'HIGH'
    """
    df_input = pd.DataFrame([input_data])

    # Derived features (must mirror MODULE 2 engineering)
    df_input["loan_income_ratio"] = (
        df_input["loan_amount"] / (df_input["annual_income"] + 1e-6)
    )
    df_input["payment_burden"] = (
        df_input["debt_to_income"] * df_input["loan_amount"]
        / (df_input["annual_income"] + 1e-6)
    )
    df_input["risk_index"] = (
        df_input["delinquencies_2yrs"] * 10
        + (850 - df_input["credit_score"]) / 85
    )

    prob = full_pipeline.predict_proba(df_input[all_features])[0, 1]
    pred = int(prob >= 0.5)
    risk = "LOW" if prob >= 0.7 else ("MEDIUM" if prob >= 0.4 else "HIGH")
    return {"prediction": pred, "probability": round(float(prob), 4), "risk_level": risk}


# Smoke test
sample = {
    "age": 35, "annual_income": 65000, "loan_amount": 12000,
    "interest_rate": 11.5, "employment_years": 4.0,
    "credit_score": 680, "debt_to_income": 22.0,
    "num_credit_lines": 8, "delinquencies_2yrs": 0,
    "loan_purpose": "debt_consolidation", "loan_grade": "B",
    "home_ownership": "RENT", "state": "CA"
}
result = predict_loan_risk(sample)
print(f"  Smoke-test result  : {result}")

# ---------------------------------------------------------------------------
# MODULE 13 — PIPELINE STATE PERSISTENCE (.sav)
# ---------------------------------------------------------------------------
print("\n[MODULE 13] Persisting full pipeline state as .sav artifact …")

pipeline_state = {
    "pipeline":         full_pipeline,
    "predict_fn":       predict_loan_risk,
    "numeric_features": NUMERIC_FEATURES,
    "cat_features":     CATEGORICAL_FEATURES,
    "all_features":     all_features,
    "target":           TARGET,
    "metrics": {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "roc_auc":  round(roc_auc, 4)
    }
}

joblib.dump(pipeline_state, PIPELINE_PATH)
print(f"  Pipeline saved → {PIPELINE_PATH}")

print("\n" + "=" * 60)
print("  ALL 13 MODULES COMPLETE — Pipeline ready for inference")
print("=" * 60)
