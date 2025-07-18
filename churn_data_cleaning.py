# -*- coding: utf-8 -*-
"""Churn Data cleaning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vWMdf4m9iZbXlZ-CMXVPcUAo9YDVQCTS
"""

import pandas as pd
import numpy as np

def clean_data():
    df = pd.read_csv('churn.csv')
    #df.describe() shows numerical information, like the number of values, the mean and the highest and lowest value of a column
    # -> important to indicate scaling differences
    #df.describe()

    #df.info() shows the Data types, the number of non-null values and the memory usage
    # -> important to indicate missing values or categorical features
    #df.info()

    #df.head() shows me the first five rows of the dataset, it is raisable
    # -> important to indicate unrealistic values
    #df.head()

    #df.duplicated().sum() shows possible duplicated rows
    # -> important to avoid incorrect weighting
    #df.duplicated().sum()

    #df['column'].unique() shows every different value in the specified column
    #df['StreamingMovies'].unique()

    #I am changing same meaning values to one general value
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
    df['InternetService'] = df['InternetService'].replace('DSL', 'Yes')
    df['InternetService'] = df['InternetService'].replace('Fiber optic', 'Yes')
    df['OnlineSecurity'] = df['OnlineSecurity'].replace('No internet service', 'No')
    df['OnlineBackup'] = df['OnlineBackup'].replace('No internet service', 'No')
    df['DeviceProtection'] = df['DeviceProtection'].replace('No internet service', 'No')
    df['TechSupport'] = df['TechSupport'].replace('No internet service', 'No')
    df['StreamingTV'] = df['StreamingTV'].replace('No internet service', 'No')
    df['StreamingMovies'] = df['StreamingMovies'].replace('No internet service', 'No')
    df['Contract'] = df['Contract'].replace('One year', 'fixed')
    df['Contract'] = df['Contract'].replace('Two year', 'fixed')
    df['Contract'] = df['Contract'].replace('Month-to-month', 'flexible')
    df['PaymentMethod'] = df['PaymentMethod'].replace('Electronic check', 'check')
    df['PaymentMethod'] = df['PaymentMethod'].replace('Mailed check', 'check')
    df['PaymentMethod'] = df['PaymentMethod'].replace('Bank transfer (automatic)', 'automatic')
    df['PaymentMethod'] = df['PaymentMethod'].replace('Credit card (automatic)', 'automatic')

    #Any vlaue that is not a float becomes NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    #Every row in the tenure-column with the value zero gets deleted
    #tenure = 0 are new customer, which are less than a month customer
    #they have no value for Totalcharges and are therefore ignored
    df = df[~df['TotalCharges'].isna() | (df['tenure'] == 0)]
    #I give every new customer with no value the number 0 for TotalCharges
    df.loc[(df['tenure'] == 0) & (df['TotalCharges'].isna()), 'TotalCharges'] = 0.0

    return df