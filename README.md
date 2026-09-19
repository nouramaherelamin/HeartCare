# ❤️ HeartCare AI

Explainable Machine Learning for Heart Disease Risk Prediction

HeartCare AI is an end-to-end machine learning project that combines predictive modeling, feature engineering, model evaluation, explainability, threshold analysis, and an interactive Streamlit application.

---

## 🚀 Overview

HeartCare AI is designed to analyze clinical input features and generate a model-based probability of heart disease.

The project goes beyond a basic machine learning model by providing an interactive platform for:

- Individual predictions
- Batch predictions
- Model evaluation
- Threshold analysis
- Error analysis
- Explainability
- Patient comparison
- Model monitoring
- Data exploration
- Prediction history
- Downloadable prediction reports

> **Important:** HeartCare AI is an educational and research project. Its output is a model-generated probability, not a medical diagnosis or clinical recommendation.

---

## 🎯 Project Goals

The main goals of this project are to:

- Build a complete machine learning pipeline.
- Apply feature engineering to improve the input representation.
- Train and tune a Logistic Regression model.
- Evaluate the model using multiple performance metrics.
- Analyze the effect of the decision threshold.
- Provide interpretable model outputs.
- Deploy the trained model through an interactive web application.

---

## 🧠 Machine Learning Workflow

### Raw Dataset

↓ ### Data Cleaning ↓ ### Feature Engineering ↓ Preprocessing ↓ ### Tuned Logistic Regression ↓ ### Threshold Optimization ↓ ### Model Evaluation ↓ ### Streamlit Deployment

---

## 📊 Model Performance

The final trained model is a **Tuned Logistic Regression** model.

| Metric      |  Score |
| ----------- | -----: |
| Accuracy    | 86.89% |
| Precision   | 81.25% |
| Recall      | 92.86% |
| F1 Score    | 86.67% |
| ROC-AUC     | 96.43% |
| PR-AUC      | 96.08% |
| Brier Score | 0.0803 |

The configured decision threshold is approximately **0.39**.

---

## ⚙️ Feature Engineering

The project creates additional features to provide the model with richer representations of the original clinical variables.

### Engineered Features

- age_group
- thalach_age_ratio
- chol_age_ratio
- bp_age_ratio
- oldpeak_thalach_ratio
- exercise_stress_index
- cardio_burden

The project metadata records the final selected feature set as:

thalach thal cp ca oldpeak

---

## 🔍 Explainable AI

HeartCare AI does not only return a prediction.

The application provides model-focused explanations such as:

- Model-generated probability
- Decision threshold
- Feature contributions
- Global model insights
- Prediction sensitivity
- Error analysis

This helps users understand how the model behaves instead of treating its output as an unexplained result.

---

## 🎚️ Threshold Analysis

The application includes a dedicated Threshold Lab for exploring how changing the decision threshold affects model performance.

The analysis includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- Confusion Matrix
- Threshold performance curves

The project uses an optimized threshold of approximately **0.39**.

---

## 🔎 Error Analysis

The application provides a dedicated error analysis section to inspect:

- Correct predictions
- False Positives
- False Negatives
- Prediction distributions
- Detailed prediction records

This makes it possible to study where the model succeeds and where it makes mistakes.

---

## 📈 Model Evaluation

HeartCare AI includes an evaluation dashboard with:

- **ROC** Curve
- Precision-Recall Curve
- Confusion Matrix
- Calibration analysis
- Classification metrics
- Probability analysis

The evaluation artifacts are stored separately from the application code.

---

## 👥 Patient Comparison

The Patient Comparison feature allows users to compare model-generated probabilities for different patient inputs.

This provides an interactive way to explore how changing input features can affect model output.

---

## 📁 Batch Prediction

Users can upload a **CSV** file containing the required input features and generate predictions for multiple records.

The workflow includes:

Upload **CSV** ↓ ### Validate Input ↓ ### Run Model ↓ ### Generate Probabilities ↓ ### Generate Predictions ↓ ### Download Results

---

## 📊 Data Explorer

The application includes an interactive dataset explorer for:

- Dataset preview
- Statistical summaries
- Target distribution
- Feature distributions
- Correlation analysis

---

## 🛡️ Model Monitoring

HeartCare AI also includes model monitoring capabilities to help analyze changes between reference data and current prediction inputs.

The monitoring layer includes distribution-based analysis such as:

- **PSI**
- KS-based analysis
- Categorical distribution checks

---

## 📝 Model Card

The project includes a dedicated Model Card section documenting important information about:

- Model type
- Dataset
- Features
- Threshold
- Evaluation metrics
- Training configuration
- Deployment information
- Limitations

---

## 🖥️ Application

The project is deployed as an interactive Streamlit application with a professional dark red interface.

### Main Sections

- Home
- Prediction
- Batch Prediction
- Data Explorer
- Threshold Lab
- Evaluation Studio
- Explainability
- Patient Comparison
- Model Card
- Model Monitor
- Model Insights
- Model Validation
- Error Analysis
- Prediction History
- About

---

## 🛠️ Tech Stack

### Programming & Data

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Logistic Regression
- Feature Engineering
- Model Evaluation

### Deployment & Visualization

- Streamlit
- Plotly
- Joblib

### Development

- Jupyter Notebook
- Git
- GitHub

---

## 📂 Project Structure

HeartCare-AI/

├── app.py ├── heart_disease_clean.csv ├── Heart_Disease_Prediction.ipynb ├── **README**.md ├── requirements.txt ├── .gitignore │ └── artifacts/ ├── final_pipeline.pkl ├── feature_metadata.json ├── model_metrics.json └── predictions.csv

---

## ⚡ Installation

### 1. Clone the repository

git clone YOUR_REPOSITORY_URL

cd heartcare-ai

### 2. Create a virtual environment

python -m venv .venv

### 3. Activate the environment

#### Windows

.venv\Scripts\activate

#### macOS / Linux

source .venv/bin/activate

### 4. Install dependencies

pip install -r requirements.txt

---

## ▶️ Run the Application

Start the Streamlit application:

python -m streamlit run app.py

The application will open in your browser.

---

## 📌 Dataset

The project uses a cleaned heart disease dataset containing the model input variables and target variable.

The application expects the dataset and trained artifacts to be available in the project structure described above.

---

## 📦 Model Artifacts

The artifacts directory contains the files required by the deployed application:

| File                  | Purpose                           |
| --------------------- | --------------------------------- |
| final_pipeline.pkl    | Trained ML pipeline               |
| feature_metadata.json | Model and feature configuration   |
| model_metrics.json    | Recorded evaluation metrics       |
| predictions.csv       | Evaluation and prediction records |

---

## ⚠️ Disclaimer

HeartCare AI is developed for **educational and research purposes**.

The model output represents a **machine learning prediction/probability** and should not be interpreted as a medical diagnosis.

The application should not be used to make clinical decisions or replace professional medical evaluation.

---

## 👤 Author
<div align="center">
  
**Noura Maher Elamin**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/nouramaherelamin/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/nouramaherelamin)

---

❤️ Built with Python, Scikit-learn & Streamlit

© **2026** Noura Maher Elamin. All Rights Reserved.
