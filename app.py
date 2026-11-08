import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

st.set_page_config(page_title="Steel Plate Fault Lab", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl",
}


@st.cache_resource
def load_artifacts(model_name):
    return (joblib.load(MODEL_FILES[model_name]),
            joblib.load("model/scaler.pkl"),
            joblib.load("model/meta.pkl"))


st.title("Steel Plate Fault Classification")
st.caption("Compare five classifiers on the UCI Steel Plates Faults data.")

# ---- Sidebar ------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    st.markdown("The CSV needs the 27 feature columns and a `Fault` column "
                "(use `test_data.csv` from the repo).")

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
model, scaler, meta = load_artifacts(model_name)
features, target = meta["features"], meta["target"]

missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X = scaler.transform(df[features])
pred = model.predict(X)

tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion matrix", "Data preview"])

if target not in df.columns:
    st.warning(f"No `{target}` column found, so only predictions are shown.")
    out = df.copy()
    out["Predicted"] = pred
    st.dataframe(out)
    st.stop()

y = df[target]
proba = model.predict_proba(X)

with tab1:
    st.subheader(f"Results: {model_name}")
    c = st.columns(6)
    c[0].metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")
        c[1].metric("AUC", f"{auc:.3f}")
    except ValueError:
        c[1].metric("AUC", "n/a")
    c[2].metric("Precision", f"{precision_score(y, pred, average='weighted', zero_division=0):.3f}")
    c[3].metric("Recall", f"{recall_score(y, pred, average='weighted', zero_division=0):.3f}")
    c[4].metric("F1", f"{f1_score(y, pred, average='weighted', zero_division=0):.3f}")
    c[5].metric("MCC", f"{matthews_corrcoef(y, pred):.3f}")
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(classification_report(
        y, pred, output_dict=True, zero_division=0)).T.round(3))

with tab2:
    labels = sorted(y.unique())
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=labels),
                           display_labels=labels).plot(ax=ax, xticks_rotation=45,
                                                       colorbar=False)
    st.pyplot(fig)

with tab3:
    st.dataframe(df.head(50))
