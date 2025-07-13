# Customer Churn Classifier

This project implements a machine learning classifier to predict customer churn in a telecom scenario based on structured customer data. The goal is to compare multiple classification models in terms of their performance on the same dataset.

## Project Overview

- **Type**: Binary classification (churn prediction)
- **Dataset**: CSV file with customer features such as contract type, tenure, payment method, etc.
- **Language**: Python
- **Libraries**: NumPy, pandas, scikit-learn

## Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Workflow

### Data Preprocessing

- Cleaning inconsistent or redundant categories
- Type conversions and handling of missing values
- Feature grouping and categorical reduction (e.g. payment methods, contract types)

### Model Training

- Train/test split
- Structured pipelines using `sklearn.pipeline.Pipeline`
- Hyperparameter optimization via `GridSearchCV`

### Model Evaluation

- Precision
- Recall
- F1-Score
- ROC AUC Score

### Result Comparison

- Evaluate and compare all models using consistent performance metrics
- Identify the most reliable classifier for churn detection

## Folder Structure (Example)

churn-prediction/
├── data/
│   └── churn.csv
├── scripts/
│   ├── churn_data_cleaning.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   └── random_forest.py
├── main.py
└── README.md

## Goal

The aim of this project was to develop a robust baseline model for business-relevant classification tasks using standard ML tools. 
To improve code clarity and model reproducibility, structured **pipelines** and **GridSearchCV** were used.
