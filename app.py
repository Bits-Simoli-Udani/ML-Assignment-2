"""
app.py - Streamlit Web Application for BITS Pilani ML Assignment 2
==================================================================
M.Tech (AIML) - Machine Learning Assignment 2
Student Coursework Submission

Features Implemented:
1. Dataset Upload Option (CSV) + Quick-Load Preloaded Test Dataset
2. Model Selection Dropdown (All 5 Classifiers + Multi-Model Comparison)
3. Display of All 6 Evaluation Metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. Visual Confusion Matrix (Heatmap) & Interactive Classification Report
5. Comparative Benchmark & ROC Curves
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 | Classification & Evaluation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic, polished styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #DBEAFE;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths & Metadata Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_TEST_CSV = os.path.join(BASE_DIR, "test_data.csv")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

@st.cache_resource
def load_artifacts():
    """Loads metadata, scaler, and all 5 pre-trained models."""
    if not os.path.exists(METADATA_FILE):
        return None, None, None

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    models = {}
    for name, config in metadata["models"].items():
        model_path = os.path.join(MODEL_DIR, config["filename"])
        if os.path.exists(model_path):
            models[name] = {
                "model": joblib.load(model_path),
                "use_scaled": config["use_scaled"]
            }

    return metadata, scaler, models

metadata, scaler, trained_models = load_artifacts()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown('<div class="main-title">🔬 Machine Learning Assignment - 2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>BITS Pilani - M.Tech (AIML)</b> | End-to-End Classification, Evaluation & Model Comparison Dashboard</div>', unsafe_allow_html=True)

if metadata is None or trained_models is None:
    st.error("⚠️ Trained models or metadata not found. Please run `python train.py` first to generate models and test data.")
    st.stop()

# ---------------------------------------------------------
# Sidebar: Data Source & Model Selector
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

# Feature: Dataset upload option (CSV) [Rubric Item a]
st.sidebar.subheader("1. Test Dataset Source")
data_source = st.sidebar.radio(
    "Select Test Data Input:",
    ["Use Preloaded Test Dataset (test_data.csv)", "Upload Custom Test CSV"],
    index=0
)

uploaded_df = None
if data_source == "Use Preloaded Test Dataset (test_data.csv)":
    if os.path.exists(DEFAULT_TEST_CSV):
        uploaded_df = pd.read_csv(DEFAULT_TEST_CSV)
        st.sidebar.success(f"✓ Loaded default `test_data.csv` ({len(uploaded_df)} samples)")
    else:
        st.sidebar.error("`test_data.csv` was not found in project directory.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Test CSV (with features + target)", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✓ Uploaded `{uploaded_file.name}` ({len(uploaded_df)} samples)")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

# Option to download test_data.csv sample
if os.path.exists(DEFAULT_TEST_CSV):
    with open(DEFAULT_TEST_CSV, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download sample `test_data.csv`",
            data=f,
            file_name="sample_test_data.csv",
            mime="text/csv"
        )

# Feature: Model Selection Dropdown [Rubric Item b]
st.sidebar.subheader("2. Model Selection")
model_options = list(trained_models.keys()) + ["Compare All 5 Models"]
selected_model_option = st.sidebar.selectbox(
    "Choose Classifier to Evaluate:",
    model_options,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Assignment Details:**
    - **Dataset:** Breast Cancer Wisconsin
    - **Features:** 30 continuous attributes
    - **Instances:** 569 instances
    - **Target:** 0: Malignant, 1: Benign
    """
)

# ---------------------------------------------------------
# Main Panel: Validation & Data Preview
# ---------------------------------------------------------
if uploaded_df is None:
    st.info("👆 Please select or upload a test dataset from the sidebar to view evaluation results.")
    st.stop()

# Validate features and target
expected_features = metadata["features"]
target_col = metadata["target_col"]

missing_features = [f for f in expected_features if f not in uploaded_df.columns]
has_target = target_col in uploaded_df.columns

if missing_features:
    st.error(f"❌ Uploaded CSV is missing {len(missing_features)} required feature(s): {missing_features[:5]}...")
    st.stop()

if not has_target:
    st.error(f"❌ Uploaded CSV must contain the ground-truth target column: `{target_col}`.")
    st.stop()

X_eval = uploaded_df[expected_features]
y_eval = uploaded_df[target_col]

with st.expander("📊 Test Dataset Summary & Preview", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Test Instances", len(uploaded_df))
    col_b.metric("Number of Features", len(expected_features))
    class_counts = y_eval.value_counts().to_dict()
    mal_count = class_counts.get(0, 0)
    ben_count = class_counts.get(1, 0)
    col_c.metric("Class Ratio (Mal / Ben)", f"{mal_count} / {ben_count}")
    st.dataframe(uploaded_df.head(5), use_container_width=True)

# Pre-scale data for models that need scaling
X_eval_scaled = scaler.transform(X_eval) if scaler is not None else X_eval

# Helper function to compute all 6 metrics
def compute_metrics(clf, use_scaled, X_raw, X_sc, y_true):
    X_input = X_sc if use_scaled else X_raw
    y_pred = clf.predict(X_input)

    # Calculate probabilities if available
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_input)[:, 1]
    elif hasattr(clf, "decision_function"):
        df_scores = clf.decision_function(X_input)
        y_prob = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-8)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

# ---------------------------------------------------------
# View 1: Single Model Evaluation
# ---------------------------------------------------------
if selected_model_option != "Compare All 5 Models":
    model_name = selected_model_option
    model_info = trained_models[model_name]
    clf = model_info["model"]
    use_scaled = model_info["use_scaled"]

    # Compute metrics without retraining [Rubric Requirement]
    eval_res = compute_metrics(clf, use_scaled, X_eval, X_eval_scaled, y_eval)

    st.markdown(f"### 🎯 Evaluation Results: `{model_name}`")
    st.caption("Evaluated directly on the uploaded test set without retraining.")

    # Feature: Display of 6 Evaluation Metrics [Rubric Item c]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{eval_res['Accuracy']:.4f}")
    m2.metric("AUC Score", f"{eval_res['AUC']:.4f}")
    m3.metric("Precision", f"{eval_res['Precision']:.4f}")
    m4.metric("Recall", f"{eval_res['Recall']:.4f}")
    m5.metric("F1 Score", f"{eval_res['F1']:.4f}")
    m6.metric("MCC Score", f"{eval_res['MCC']:.4f}")

    st.markdown("---")

    # Feature: Confusion Matrix or Classification Report [Rubric Item d]
    tab1, tab2, tab3 = st.tabs(["📉 Confusion Matrix", "📋 Classification Report", "📈 ROC Curve"])

    with tab1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_eval, eval_res["y_pred"])
        
        fig, ax = plt.subplots(figsize=(5, 3.8))
        labels = ["Malignant (0)", "Benign (1)"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
            cbar=False, annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
        col_cm1.success(f"True Positive (Benign): **{tp}**")
        col_cm2.info(f"True Negative (Malignant): **{tn}**")
        col_cm3.warning(f"False Positive (Type I): **{fp}**")
        col_cm4.error(f"False Negative (Type II): **{fn}**")

    with tab2:
        st.subheader("Detailed Classification Report")
        target_names = ["Malignant (0)", "Benign (1)"]
        report_dict = classification_report(
            y_eval, eval_res["y_pred"], target_names=target_names, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(
            report_df.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}"
            }),
            use_container_width=True
        )

    with tab3:
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        if len(np.unique(y_eval)) > 1:
            fpr, tpr, _ = roc_curve(y_eval, eval_res["y_prob"])
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {eval_res['AUC']:.4f})")
            ax_roc.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
            ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
            ax_roc.set_title(f"ROC Curve: {model_name}", fontsize=11, fontweight="bold")
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        else:
            st.warning("ROC Curve requires both classes to be present in the test set.")

# ---------------------------------------------------------
# View 2: Multi-Model Comparison (Step 5 Comparison Table)
# ---------------------------------------------------------
else:
    st.markdown("### 🏆 Comprehensive Model Comparison Table")
    st.caption("Direct evaluation of all 5 models on the identical test set:")

    comparison_records = []
    for m_name, m_info in trained_models.items():
        res = compute_metrics(m_info["model"], m_info["use_scaled"], X_eval, X_eval_scaled, y_eval)
        comparison_records.append({
            "ML Model Name": m_name,
            "Accuracy": res["Accuracy"],
            "AUC": res["AUC"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1": res["F1"],
            "MCC": res["MCC"]
        })

    comp_df = pd.DataFrame(comparison_records)

    # Display comparison table matching Step 5 format
    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#C7D2FE"
        ).format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True
    )

    # Visual Metric Comparison Chart
    st.subheader("📊 Metric Comparison Across Models")
    chart_metric = st.selectbox(
        "Select Metric to Visualize Across Models:",
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        index=4
    )

    fig_bar, ax_bar = plt.subplots(figsize=(7, 3.8))
    palette = sns.color_palette("Blues_r", n_colors=len(comp_df))
    bars = ax_bar.bar(comp_df["ML Model Name"], comp_df[chart_metric], color=palette)
    ax_bar.set_ylim([0.8, 1.02])
    ax_bar.set_ylabel(chart_metric, fontweight="bold")
    ax_bar.set_title(f"{chart_metric} Comparison Across All 5 Models", fontsize=11, fontweight="bold")
    plt.xticks(rotation=20, ha="right")

    for bar in bars:
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.005,
            f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # Highlight Overall Winner
    winner_row = comp_df.sort_values(by=["F1", "AUC", "Accuracy", "MCC"], ascending=False).iloc[0]
    st.success(
        f"🎖️ **Overall Winner for this Dataset:** **{winner_row['ML Model Name']}** "
        f"(F1: {winner_row['F1']:.4f}, AUC: {winner_row['AUC']:.4f}, MCC: {winner_row['MCC']:.4f})"
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem;">
        <b>BITS Pilani - Work Integrated Learning Programmes Division</b> | M.Tech (AIML) Machine Learning Assignment 2<br>
        Models serialized with <code>scikit-learn</code> and <code>joblib</code>. Frontend rendered via <code>Streamlit</code>.
    </div>
    """,
    unsafe_allow_html=True
)
