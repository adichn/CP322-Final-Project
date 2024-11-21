import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read the data

data = pd.read_csv("student-mat.csv", sep=";")

# Data Quality Report

def generate_data_quality_report(df):


    print("====== Data Quality Report ======")
    
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Missing values analysis

    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")
    
    # Data types

    print("\nData Types:\n", df.dtypes)
    
    # Summary statistics for numerical columns

    print("\nNumerical Columns Summary:\n", df.describe())
    
    # Categorical columns analysis

    categorical_cols = df.select_dtypes(include=['object']).columns
    print("\nCategorical Columns Analysis:")
    for col in categorical_cols:
        print(f"\nUnique values in {col}:\n", df[col].value_counts())
        
    return df

# Create basic visualizations

def create_basic_visualizations(df):

    # Set up the figure

    plt.figure(figsize=(15, 10))
    
    # Grade distribution

    plt.subplot(2, 2, 1)
    plt.hist(df['G3'].values, bins=20)
    plt.title('Final Grade Distribution')
    plt.xlabel('Final Grade')
    plt.ylabel('Count')
    
    # Study time vs final grade

    plt.subplot(2, 2, 2)

    # Calculate mean grades for each study time

    study_time_means = []
    study_times = sorted(df['studytime'].unique())
    for st in study_times:
        mean_grade = df[df['studytime'] == st]['G3'].mean()
        study_time_means.append(mean_grade)
    plt.bar(study_times, study_time_means)
    plt.title('Average Grade by Study Time')
    plt.xlabel('Study Time')
    plt.ylabel('Average Final Grade')
    
    # Mother's education vs final grade

    plt.subplot(2, 2, 3)

    # Calculate mean grades for each mother's education level

    medu_means = []
    medu_levels = sorted(df['Medu'].unique())
    for edu in medu_levels:
        mean_grade = df[df['Medu'] == edu]['G3'].mean()
        medu_means.append(mean_grade)
    plt.bar(medu_levels, medu_means)
    plt.title("Average Grade by Mother's Education")
    plt.xlabel("Mother's Education Level")
    plt.ylabel('Average Final Grade')
    
    # Failures vs final grade

    plt.subplot(2, 2, 4)

    # Calculate mean grades for each number of failures

    failure_means = []
    failure_counts = sorted(df['failures'].unique())
    for f in failure_counts:
        mean_grade = df[df['failures'] == f]['G3'].mean()
        failure_means.append(mean_grade)
    plt.bar(failure_counts, failure_means)
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
    
    # Split features and target, removing intermediate grades and setting Final grade as target

    X = df_encoded.drop(['G1', 'G2', 'G3'], axis=1)
    y = df_encoded['G3']
    
    # Split data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# Additional analysis functions

def correlation_analysis(df):

    # Select numerical columns

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlations = df[numerical_cols].corr()
    
    # Print correlations with G3 (final grade)

    print("\nCorrelations with Final Grade (G3):")
    print(correlations['G3'].sort_values(ascending=False))
    
    return correlations

# Run the analysis

try:

    # Generate data quality report

    df = generate_data_quality_report(data)
    
    # Perform correlation analysis

    correlations = correlation_analysis(df)
    
    # Create visualizations

    create_basic_visualizations(df)
    
    # Prepare the data for modeling

    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print("\nData preparation completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
except Exception as e:
    
    print(f"An error occurred: {str(e)}")