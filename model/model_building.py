# =====================================
# BREAST CANCER MODEL DEVELOPMENT
# =====================================

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 0 = malignant, 1 = benign

# -------------------------------
# Feature Selection (5 only)
# -------------------------------
selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean concavity'
]

X = X[selected_features]

# -------------------------------
# Check Missing Values
# -------------------------------
print(X.isnull().sum())

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Feature Scaling (MANDATORY)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# -------------------------------
# Save Model & Scaler
# -------------------------------
joblib.dump(model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")

# -------------------------------
# Reload Test (Required by guideline)
# -------------------------------
loaded_model = joblib.load("breast_cancer_model.pkl")
test_prediction = loaded_model.predict(X_test_scaled[:1])
print("Reloaded model prediction successful:", test_prediction)
