import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read Data

data = pd.read_csv("student-mat.csv", sep=";")
# print(data)  # Checking if it is loaded

# Data Quality Report

def generate_data_quality_report(df):
    print("\n")
    print("====== Data Quality Report ======")
    print("\n")

    # Basic Information

    print("\n")
    print("Dataset Shape: ", df.shape)
    print("\n")
    print("Columns: ", df.columns.tolist())
    print("\n")

    # Missing Values Analysis

    missing_values = df.isnull().sum()
    print("\n")
    print("Missing Values: \n", missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")
    print("\n")

    # Data Types

    print("\n")
    print("Data Types: \n", df.dtypes)
    print("\n")

    # Summary Statistics for Numerical Columns 

    print("\n")
    print("Numerical Columns Summary: \n", df.describe())
    print("\n")

    # Categorical Columns Analysis

    categorial_cols = df.sekect_dtypes(include='object').columns
    print("\n")
    print("Categorical Columns Analysis: \n")
    for col in categorial_cols:
        print(f"Unique values in {col}: \n", df[col].value_counts())

    return df


    