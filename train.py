"""
train.py - Model Training & Evaluation Pipeline for BITS ML Assignment 2
======================================================================
Course: M.Tech (AIML) - Machine Learning
Assignment: Assignment 2 (15 Marks)

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / Kaggle)
- Total Instances: 569 (Satisfies >= 500 requirement)
- Total Features: 30 continuous numeric features (Satisfies >= 12 requirement)
- Problem Type: Binary Classification (0: Malignant, 1: Benign)

Models Implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (KNN)
4. Naive Bayes Classifier (GaussianNB)
5. Support Vector Machine (SVM)

Metrics Computed:
1. Accuracy
2. AUC Score (ROC-AUC)
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

def main():
    print("=" * 70)
    print(" BITS Pilani M.Tech (AIML) - ML Assignment 2 Model Training Pipeline")
    print("=" * 70)

    # 1. Base directory setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # 2. Load Dataset
    print("\n[Step 1/5] Loading Breast Cancer Wisconsin (Diagnostic) dataset...")
    data = load_breast_cancer()
    feature_names = [f.replace(" ", "_") for f in data.feature_names]
    
    df = pd.DataFrame(data.data, columns=feature_names)
    df["target"] = data.target  # 0 = Malignant, 1 = Benign

    print(f"-> Total Instances: {df.shape[0]} (Requirement: >= 500)")
    print(f"-> Total Features : {len(feature_names)} (Requirement: >= 12)")
    print(f"-> Target Distribution: Malignant (0): {(df['target'] == 0).sum()}, Benign (1): {(df['target'] == 1).sum()}")

    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    print("\n[Step 2/5] Creating stratified 80/20 train-test split...")
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"-> Train samples: {len(X_train)}")
    print(f"-> Test samples : {len(X_test)}")

    # Save test_data.csv (features + target) for Streamlit upload and validation
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_csv_path = os.path.join(base_dir, "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"-> Exported hold-out test set to: {test_csv_path}")

    # 4. Feature Standardization
    print("\n[Step 3/5] Fitting StandardScaler on training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"-> Saved feature scaler to: {scaler_path}")

    # 5. Define All 5 Classifiers
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            "use_scaled": True,
            "filename": "logistic_regression.joblib"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
            "use_scaled": False,
            "filename": "decision_tree.joblib"
        },
        "KNN": {
            "model": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
            "use_scaled": True,
            "filename": "knn.joblib"
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "use_scaled": False,
            "filename": "naive_bayes.joblib"
        },
        "Support Vector Machine": {
            "model": SVC(kernel="rbf", probability=True, C=1.0, random_state=42),
            "use_scaled": True,
            "filename": "svm.joblib"
        }
    }

    # 6. Train, Evaluate & Serialize Models
    print("\n[Step 4/5] Training models and computing the 6 required evaluation metrics...")
    results = []
    metadata_metrics = {}

    for name, config in models.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]
        filename = config["filename"]

        # Select data representation
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        # Train model
        clf.fit(X_tr, y_train)

        # Inference on test set
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # Calculate all 6 metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Save model
        save_path = os.path.join(model_dir, filename)
        joblib.dump(clf, save_path)

        metrics_dict = {
            "Accuracy": round(float(acc), 4),
            "AUC": round(float(auc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "MCC": round(float(mcc), 4),
        }

        results.append({
            "ML Model Name": name,
            **metrics_dict
        })

        metadata_metrics[name] = {
            "metrics": metrics_dict,
            "use_scaled": use_scaled,
            "filename": filename
        }

        print(f"  [OK] {name:<24} | Acc: {acc:.4f} | AUC: {auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

    # 7. Save metadata.json for the Streamlit App
    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic) Dataset",
        "source": "UCI Machine Learning Repository / Kaggle",
        "features": feature_names,
        "target_col": "target",
        "target_classes": {0: "Malignant", 1: "Benign"},
        "models": metadata_metrics
    }
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n[Step 5/5] Exported model metadata to: {metadata_path}")

    # 8. Print Assignment Comparison Table
    print("\n" + "=" * 80)
    print("ASSIGNMENT COMPARISON TABLE (Section 3 - Step 5):")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 80)

    # Determine Winner
    winner_row = results_df.sort_values(by=["F1", "AUC", "Accuracy", "MCC"], ascending=False).iloc[0]
    print(f"\nOVERALL WINNER: {winner_row['ML Model Name']}")
    print(f"Details: F1={winner_row['F1']:.4f}, AUC={winner_row['AUC']:.4f}, Accuracy={winner_row['Accuracy']:.4f}, MCC={winner_row['MCC']:.4f}\n")

if __name__ == "__main__":
    main()
