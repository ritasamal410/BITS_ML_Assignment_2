"""
Train the five classifiers required by the BITS M.Tech AIML ML Assignment 2.

Run this on BITS Virtual Lab:
    python -m pip install -r requirements.txt
    python train_models.py

The script downloads the UCI Dry Bean dataset, creates a stratified 80:20 split,
trains all five required models, saves fitted pipelines, writes test_data.csv,
and creates metrics_summary.csv. Results are calculated from the held-out test set.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Class"

def fetch_data():
    print("Fetching UCI Dry Bean dataset (UCI dataset ID: 602)...")
    dataset = fetch_ucirepo(id=602)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    frame = X.copy()
    frame[TARGET] = y.astype(str).values
    frame = frame.dropna()
    return frame

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    data = fetch_data()
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(str)

    print(f"Dataset shape: {data.shape}; input features: {X.shape[1]}; rows: {len(data)}")
    if X.shape[1] < 12 or len(data) < 500:
        raise ValueError("Dataset does not meet the assignment minimum of 12 features and 500 rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Standardization is included for distance- and margin-based models.
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, class_weight=None))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, min_samples_leaf=2
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ])
    }

    # Persist the test split so the Streamlit app can evaluate the saved models.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.values
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    classes = sorted(y.unique())
    for display_name, estimator in models.items():
        print(f"Training {display_name} ...")
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)

        metrics = {
            "Model": display_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro", labels=estimator.classes_),
            "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "F1": f1_score(y_test, predictions, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions)
        }
        rows.append(metrics)
        safe_name = display_name.lower().replace(" ", "_").replace("-", "")
        joblib.dump(estimator, MODEL_DIR / f"{safe_name}.joblib")
        print("  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "Model"))

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    results.to_csv(ROOT / "metrics_summary.csv", index=False)
    metadata = {
        "dataset": "UCI Dry Bean",
        "uci_dataset_id": 602,
        "target": TARGET,
        "feature_columns": list(X.columns),
        "classes": classes,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }
    (ROOT / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nHeld-out test-set comparison:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCreated test_data.csv, metrics_summary.csv, model_metadata.json and models/*.joblib")
    print("Review the output, then write your own observations in README.md before submission.")

if __name__ == "__main__":
    main()
