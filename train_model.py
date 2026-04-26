import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Starting training process... Please wait.")

# 1. Load Data
df = pd.read_csv("loan_dataset_20000.csv")
df.drop_duplicates(inplace=True)

# 2. Data Cleaning (Safely handled for latest Pandas version)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# 3. Setup X and Y
X = df.drop('loan_paid_back', axis=1)
y = df['loan_paid_back']

num_features = X.select_dtypes(include=['float64', 'int64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# 4. Create Preprocessor Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Training the Random Forest model...")
pipeline.fit(X_train, y_train)

# 6. Save Model to your local machine
os.makedirs('Models', exist_ok=True)
joblib.dump(pipeline, 'Models/full_pipeline.sav')

print("✅ Success! Fresh 'full_pipeline.sav' has been created for your system versions.")