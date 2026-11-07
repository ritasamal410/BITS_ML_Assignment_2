from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🌱", layout="wide")
st.title("🌱 Dry Bean Variety Classification")
st.write(
    "Upload a labelled test CSV, select one of the five trained classifiers, "
    "and inspect its evaluation metrics. The app loads saved models; it does not retrain."
)

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded

@st.cache_data
def load_metadata():
    path = ROOT / "model_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}

models = load_models()
metadata = load_metadata()

if not models:
    st.error("No saved models found. Run `python train_models.py` first and commit the models/ folder.")
    st.stop()

st.sidebar.header("Evaluation settings")
selected_model = st.sidebar.selectbox("Choose classifier", list(models.keys()))
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

st.caption("CSV requirement: include all 16 model input columns and the target column named `Class`.")
if uploaded is None:
    st.info("For a quick check, use the `test_data.csv` created by train_models.py.")
    default_path = ROOT / "test_data.csv"
    if default_path.exists() and st.button("Load project test_data.csv"):
        data = pd.read_csv(default_path)
    else:
        st.stop()
else:
    data = pd.read_csv(uploaded)

if TARGET not in data.columns:
    st.error("The uploaded CSV must include the target column named `Class`.")
    st.stop()

expected_features = metadata.get("feature_columns", [])
if not expected_features:
    st.error("model_metadata.json is missing. Run train_models.py and commit the generated file.")
    st.stop()

missing = [col for col in expected_features if col not in data.columns]
extra = [col for col in data.columns if col not in expected_features + [TARGET]]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()
if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

X_test = data[expected_features].apply(pd.to_numeric, errors="coerce")
if X_test.isna().any().any():
    st.error("Input features contain missing or non-numeric values. Please check the CSV.")
    st.stop()
y_true = data[TARGET].astype(str)

model = models[selected_model]
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Use model class ordering for multiclass one-vs-rest macro AUC.
try:
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=model.classes_)
except ValueError as exc:
    auc = None
    st.warning(f"AUC could not be calculated for this upload: {exc}")

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC (macro OVR)": auc,
    "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

st.subheader(f"Results: {selected_model}")
cols = st.columns(3)
for i, (label, value) in enumerate(metrics.items()):
    display_value = "Not available" if value is None else f"{value:.4f}"
    cols[i % 3].metric(label, display_value)

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    labels = list(model.classes_)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("View uploaded test data"):
    st.dataframe(data.head(100), use_container_width=True)
st.caption(f"Rows evaluated: {len(data)} | Features used: {len(expected_features)}")
