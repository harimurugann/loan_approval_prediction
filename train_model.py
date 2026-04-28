# train_model.py
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def create_synthetic_data(filename='loan_dataset_20000.csv'):
    """Generates a robust synthetic dataset for the pipeline."""
    np.random.seed(42)
    n_samples = 20000
    data = {
        'loan_amnt': np.random.uniform(1000, 40000, n_samples).round(2),
        'term': np.random.choice([36, 60], n_samples),
        'int_rate': np.random.uniform(5.0, 25.0, n_samples).round(2),
        'installment': np.random.uniform(30, 1500, n_samples).round(2),
        'annual_inc': np.random.uniform(20000, 250000, n_samples).round(2),
        'dti': np.random.uniform(0, 40, n_samples).round(2),
        'open_acc': np.random.randint(2, 25, n_samples),
        'total_acc': np.random.randint(5, 50, n_samples),
        'loan_paid_back': np.random.choice([0, 1], p=[0.2, 0.8], size=n_samples)
    }
    
    # Introduce random nulls for advanced imputation testing
    data['annual_inc'][np.random.choice(n_samples, 500, replace=False)] = np.nan
    data['dti'][np.random.choice(n_samples, 200, replace=False)] = np.nan
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Dataset '{filename}' generated successfully.")
    return filename

def main():
    # 1. Environment & Data Preparation
    csv_file = create_synthetic_data()
    df = pd.read_csv(csv_file)

    # 2. Data Integrity & Cleaning
    df['annual_inc'] = df['annual_inc'].fillna(df['annual_inc'].median())
    df['dti'] = df['dti'].fillna(df['dti'].mean())
    df = df.drop_duplicates()

    # 3. Feature-Target Architecture
    X = df.drop('loan_paid_back', axis=1)
    y = df['loan_paid_back']

    # 4. Scalable Data Modeling
    numeric_features = X.columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    # 5. Stratified Data Partitioning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6. Model Training (Champion Model)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    print("Training the model...")
    full_pipeline.fit(X_train, y_train)

    # 7. Model Evaluation
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 8. Artifact Persistence
    os.makedirs('Models', exist_ok=True)
    os.makedirs('Data', exist_ok=True)
    
    joblib.dump(clf, 'Models/champion_model.sav')
    joblib.dump(full_pipeline, 'Models/full_pipeline.sav')
    print("Pipeline state and champion model saved in 'Models/' directory.")

if __name__ == "__main__":
    main()