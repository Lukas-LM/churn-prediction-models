# Customer Churn Classifier

This project implements a machine learning classifier to predict customer churn in a telecom scenario based on structured customer data. The goal is to compare multiple classification models in terms of their performance on the same dataset.

## Project Overview

- **Type**: Binary classification (churn prediction)
- **Dataset**: Telco Customer Churn dataset from Kaggle
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
- Decision Tree: F1: ~0.51 - ROC-AUC: ~0.66
- Logistic Regression: F1: ~0.59 - ROC-AUC: ~0.71
- Random Forest: F1: ~0.58 - ROC-AUC: ~0.71

## Folder Structure (Example)
```
churn-prediction-models/
├── churn.csv
├── churn_data_cleaning.py
├── churn_decisiontree.py
├── churn_logisticregression.py
├── churn_randomforest.py
├── churn_main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```
## Goal

The aim of this project was to develop a robust baseline model for business-relevant classification tasks using standard ML tools. 
To improve code clarity and reproducibility, I implemented structured pipelines and hyperparameter tuning using GridSearchCV.
