import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

sys.path.append(os.path.dirname(__file__))
from feature_engineering import prepare_xy

warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/Ramakrishna-Kaki/Data-set-for-AI-DS/main/Electricity%20Theft%20detection%20Dataset.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("Downloading dataset...")
df = pd.read_csv(DATA_URL)
print("Dataset shape:", df.shape)

X, y, numeric, categorical = prepare_xy(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", categorical_pipe, categorical)
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}

results = []
fitted = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(X_test)[:, 1]
    else:
        prob = pred.astype(float)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob)
    }
    results.append(row)
    fitted[name] = (pipe, pred, prob)

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_pipe, best_pred, best_prob = fitted[best_name]

joblib.dump(best_pipe, f"{MODEL_DIR}/best_model.joblib")

with open(f"{MODEL_DIR}/model_info.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "target": "Theft Label",
        "selection_metric": "F1-score",
        "leakage_controls": [
            "Customer_ID removed",
            "Meter_Tampering removed"
        ]
    }, f, indent=2)

report = classification_report(y_test, best_pred, digits=4)
with open(f"{REPORT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="F1", y="Model")
plt.xlim(0, 1)
plt.title("Model Comparison — F1 Score")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/model_comparison.png", dpi=180)
plt.close()

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nBEST MODEL:", best_name)
print("\nCLASSIFICATION REPORT")
print(report)
print(f"\nSaved model to {MODEL_DIR}/best_model.joblib")
