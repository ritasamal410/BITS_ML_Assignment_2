# Dry Bean Variety Classification – Machine Learning Assignment 2

## 1. Project Overview

This project implements a multiclass machine-learning classification system to predict the variety of dry beans using numerical measurements extracted from bean images.

Five machine-learning algorithms are trained and evaluated using the same stratified training and testing dataset. Their performance is compared using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also developed to allow users to upload test datasets and evaluate the trained classification models.

## 2. Problem Statement

Classify a dry bean sample into one of seven varieties using 16 numerical features describing its shape, size and geometric properties.

The objective is to compare five classification algorithms and identify the model that provides the strongest overall classification performance.

## 3. Dataset Description

* **Dataset:** Dry Bean
* **Source:** UCI Machine Learning Repository
* **Dataset ID:** 602
* **Total instances:** 13,611
* **Input features:** 16 numerical features
* **Target column:** Class
* **Number of classes:** 7
* **Missing values:** None reported in the dataset description.

The seven bean varieties are:

1. Seker
2. Barbunya
3. Bombay
4. Cali
5. Dermason
6. Horoz
7. Sira

The dataset contains measurements such as Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, Solidity, Roundness, Compactness and ShapeFactor measurements.

**Dataset Source:** https://archive.ics.uci.edu/dataset/602/dry+bean

**Dataset DOI:** 10.24432/C50S4B

## 4. Machine Learning Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

StandardScaler is included in pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. The Decision Tree model is trained without feature scaling.

## 5. Model Training and Evaluation

The dataset was divided using a stratified train-test split.

* Training data: 80%
* Testing data: 20%
* Random state: 42
* AUC calculation: Macro One-vs-Rest
* Precision: Macro average
* Recall: Macro average
* F1 Score: Macro average
* MCC: Multiclass Matthews Correlation Coefficient

All five models were evaluated using the same test dataset.

## 6. Model Performance Comparison

The following results were obtained from the executed notebook:

| ML Model Name          |   Accuracy |        AUC |  Precision |     Recall |   F1 Score |        MCC |
| ---------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression    |     0.9207 |     0.9948 |     0.9349 |     0.9314 |     0.9329 |     0.9042 |
| Decision Tree          |     0.8884 |     0.9531 |     0.9038 |     0.9041 |     0.9036 |     0.8652 |
| K-Nearest Neighbors    |     0.9166 |     0.9833 |     0.9320 |     0.9271 |     0.9293 |     0.8992 |
| Gaussian Naive Bayes   |     0.8979 |     0.9916 |     0.9112 |     0.9092 |     0.9091 |     0.8773 |
| Support Vector Machine | **0.9221** | **0.9949** | **0.9359** | **0.9330** | **0.9343** | **0.9059** |

## 7. Model Performance Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 92.07% and an F1 score of 93.29%. It demonstrated strong classification performance and an AUC of 0.9948.

### Decision Tree

Decision Tree achieved an accuracy of 88.84% and an F1 score of 90.36%. Among the five evaluated models, it recorded the lowest accuracy and F1 score.

### K-Nearest Neighbors

KNN achieved an accuracy of 91.66% and an F1 score of 92.93%. It provided competitive performance, with an AUC of 0.9833.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 89.79% and an F1 score of 90.91%. Its AUC was 0.9916, indicating strong class discrimination despite its comparatively lower accuracy.

### Support Vector Machine

SVM achieved the highest accuracy of 92.21%, highest AUC of 0.9949 and highest F1 score of 93.43%. It also recorded the highest Precision, Recall and MCC among the five models.

### Overall Best Model: Support Vector Machine

Based on the evaluated metrics, SVM demonstrated the strongest overall performance. It achieved the highest accuracy, AUC, Precision, Recall, F1 Score and MCC in this experiment.

## 8. Streamlit Application

An interactive web application was developed using Streamlit.

The application provides the following functionality:

* Displays the performance comparison of all five trained models.
* Allows users to upload a CSV test dataset.
* Validates the uploaded feature columns and target column.
* Loads the saved trained model pipelines.
* Calculates Accuracy, AUC, Precision, Recall, F1 Score and MCC.
* Displays confusion matrices for model evaluation.
* Identifies the model with the highest macro F1 score.
* Provides an option to download evaluation metrics as a CSV file.

The application loads saved trained models and does not retrain models on uploaded test data.

## 9. Project Structure

```text
BITS_ML_Assignment_2/
│
├── data/
│   └── test_data.csv
│
├── models/
│   ├── decision_tree.joblib
│   ├── gaussian_naive_bayes.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── logistic_regression.joblib
│   └── support_vector_machine.joblib
│
├── outputs/
│   ├── metrics_summary.csv
│   └── model_metadata.json
│
├── screenshots/
│
├── ML_Assignment_2.ipynb
├── ML_Assignment_2_DryBean.ipynb
├── app.py
├── train_models.py
├── requirements.txt
└── README.md
```

## 10. Installation and Execution

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

To run the Streamlit application:

```bash
streamlit run app.py
```

To retrain the models:

```bash
python train_models.py
```

The application loads the trained model files, metadata and evaluation metrics from their respective project folders.

## 11. GitHub Repository

**Project Repository:**

https://github.com/ritasamal410/BITS_ML_Assignment_2

## 12. Streamlit Deployment

The application has been successfully tested locally using Streamlit.

**Live Streamlit Application URL:** To be added after deployme
