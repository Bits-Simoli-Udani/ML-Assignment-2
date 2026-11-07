# BITS Pilani - M.Tech (AIML)
## Machine Learning (Assignment - 2)

---

### a. Problem statement
The objective of this assignment is to implement, evaluate, and demonstrate multiple supervised machine learning classification algorithms on a real-world classification dataset. The task selected is the diagnostic classification of breast masses based on digitized Fine Needle Aspirate (FNA) cell nuclei images to predict whether a biopsy is **Malignant** (cancerous) or **Benign** (non-cancerous). 

In medical diagnostic applications, maximizing discriminatory power (AUC) and sensitivity/recall is essential to prevent false negatives, while maintaining high precision, accuracy, and Matthews Correlation Coefficient (MCC) to minimize false positives. Five distinct classifiers—Logistic Regression, Decision Tree Classifier, K-Nearest Neighbor Classifier, Naive Bayes Classifier (Gaussian), and Support Vector Machine—are trained, evaluated on an identical test split, and integrated into an interactive Streamlit web application that performs inference on uploaded test data without retraining.

---

### b. Dataset description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository (Dataset ID: 17) / Kaggle (`uciml/breast-cancer-wisconsin-data`)
- **Total Number of Instances:** 569 instances (satisfies the minimum 500 instances requirement)
- **Total Number of Features:** 30 continuous numeric features (satisfies the minimum 12 features requirement)
- **Target Column:** Binary classification (`target`):
  - `0`: Malignant (212 instances, 37.3%)
  - `1`: Benign (357 instances, 62.7%)
- **Feature Details:** Features are computed from digitized images of fine needle aspirates (FNA) of breast masses, describing characteristics of cell nuclei:
  1. *Radius* (mean of distances from center to points on the perimeter)
  2. *Texture* (standard deviation of gray-scale values)
  3. *Perimeter*
  4. *Area*
  5. *Smoothness* (local variation in radius lengths)
  6. *Compactness* ($perimeter^2 / area - 1.0$)
  7. *Concavity* (severity of concave portions of the contour)
  8. *Concave points* (number of concave portions of the contour)
  9. *Symmetry*
  10. *Fractal dimension* ("coastline approximation" - 1)
  
  For each image, the Mean, Standard Error (SE), and "Worst" (mean of the three largest values) were computed for each feature, resulting in $10 \times 3 = 30$ continuous attributes.
- **Preprocessing:** All 30 features were standardized using `StandardScaler` fitted strictly on the 80% training set (455 samples) and applied to transform the 20% hold-out test set (114 samples, saved as `test_data.csv`).

---

### c. Github Repository Link
- **GitHub Repository Link:** [https://github.com/Bits-Simoli-Udani/ML-Assignment-2](https://github.com/Bits-Simoli-Udani/ML-Assignment-2)
- **Live Streamlit App Link:** [https://simoli-bits-ml-assignment-2.streamlit.app](https://simoli-bits-ml-assignment-2.streamlit.app)

The repository contains all required files and directory structure:
```text
ML-Assignment-2/
|-- app.py                     # Streamlit web application
|-- requirements.txt           # Python dependencies
|-- README.md                  # Documentation with problem statement, dataset, table & observations
|-- test_data.csv              # Hold-out test dataset used in experiments
|-- train.py                   # Source code for data preparation and training
|-- model/                     # Saved trained models (.joblib), scaler, and source code
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- svm.joblib
    |-- scaler.joblib
    |-- train.py
    |-- metadata.json
```

---

### d. Models used

All 5 classification models were trained on the identical 80% stratified training split and evaluated on the identical 20% hold-out test split (114 samples: 42 Malignant, 72 Benign).

#### Comparison Table with evaluation metrics calculated for all 5 models:
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

---

### Observations on the performance of each model on the chosen dataset

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Logistic Regression demonstrated top-tier performance across all six metrics (**Accuracy: 0.9825, AUC: 0.9954, MCC: 0.9623**). Because the 30 continuous features were standardized, the regularized linear decision boundary separated benign and malignant nuclei with minimal classification error. Out of 114 test instances, it misclassified only 2 samples (1 False Positive, 1 False Negative). Its smooth sigmoid probability outputs are well-calibrated and clinically meaningful. |
| **Decision Tree** | Decision Tree Classifier obtained the lowest scores among the five models (**Accuracy: 0.9211, AUC: 0.9163, MCC: 0.8341**). Because decision trees split data using orthogonal axis-parallel hyperplanes, they struggle to capture complex diagonal or non-linear correlations between morphometric attributes (such as perimeter, area, and radius). Even with depth regularized to 5, the model produced 6 false negatives and 3 false positives. |
| **KNN** | K-Nearest Neighbors ($k=5$) delivered strong classification performance (**Accuracy: 0.9561, AUC: 0.9788, F1: 0.9655**). The Euclidean distance metric in the standardized 30-dimensional feature space accurately identified localized cluster neighborhoods for benign and malignant cells. However, its accuracy remains slightly lower than SVM and Logistic Regression due to minor boundary density overlap in high dimensions. |
| **Naive Bayes** | Gaussian Naive Bayes showed high discrimination (**AUC: 0.9878, Accuracy: 0.9386, F1: 0.9517**). While the core theoretical assumption of class-conditional feature independence is violated by correlated geometric measurements (e.g., area and perimeter), the model's Bayesian probabilistic ranking remains strong as reflected in its high AUC score of 0.9878. |
| **Support Vector Machine** | Support Vector Machine (RBF kernel) tied Logistic Regression for peak performance (**Accuracy: 0.9825, AUC: 0.9950, MCC: 0.9623**). The non-linear Radial Basis Function mapped the continuous standardized inputs into a higher-dimensional space where an optimal maximum-margin separating hyperplane was constructed, exhibiting high robustness against margin outliers and achieving identical precision, recall, and F1 (0.9861). |

#### Overall, Winner for your dataset?
- **Overall Winner:** **Logistic Regression** (narrowly edging out **Support Vector Machine**)
- **Justification:** While both Logistic Regression and Support Vector Machine achieved identical top values in Accuracy (0.9825), Precision (0.9861), Recall (0.9861), F1-Score (0.9861), and MCC (0.9623), **Logistic Regression** is the overall winner for this diagnostic dataset:
  1. It achieved a slightly higher Area Under the ROC Curve (**AUC: 0.9954** compared to 0.9950 for SVM).
  2. In clinical healthcare settings, Logistic Regression offers direct probabilistic calibration, allowing clinicians to interpret predicted probability scores as calibrated risk estimates.
  3. It requires fewer computational resources during inference and avoids kernel hyperparameter complexity.
