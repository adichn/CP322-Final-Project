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


# Create Basic Visualizations

def create_basic_visualizations(df):

    # Set figure up

    plt.figure(figsize = (15,10))

    #Grade Distribution

    plt.subplot(2, 2, 1)
    plt.hist(df['G3'], bins = 20)
    plt.title('Final Grade Distribution')
    plt.xlabel('Final Grade')
    plt.ylabel('Count')

    # Study Time vs Final Grade

    plt.subplot(2, 2, 2)
    plt.bar(study_grade.index, study_grade.values)
    plt.title('Average Grade by Study Time')
    plt.xlabel('Study Time')
    plt.ylabel('Average Final Grade')
    
    # Mother's education vs final grade
    plt.subplot(2, 2, 3)
    medu_grade = df.groupby('Medu')['G3'].mean()
    plt.bar(medu_grade.index, medu_grade.values)
    plt.title("Average Grade by Mother's Education")
    plt.xlabel("Mother's Education Level")
    plt.ylabel('Average Final Grade')
    
    # Failures vs final grade
    plt.subplot(2, 2, 4)
    fail_grade = df.groupby('failures')['G3'].mean()
    plt.bar(fail_grade.index, fail_grade.values)
    plt.title('Average Grade by Number of Failures')
    plt.xlabel('Number of Failures')
    plt.ylabel('Average Final Grade')
    
    plt.tight_layout()
    plt.show()

# Prepare data for modeling
def prepare_data(df):
    # Convert categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Split features and target
    X = df_encoded.drop(['G1', 'G2', 'G3'], axis=1)  # Remove intermediate grades
    y = df_encoded['G3']  # Final grade as target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# Run the analysis
try:
    # Generate data quality report
    df = generate_data_quality_report(data)
    
    # Create visualizations
    create_basic_visualizations(df)
    
    # Prepare the data for modeling
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print("\nData preparation completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
except Exception as e:
    print(f"An error occurred: {str(e)}")