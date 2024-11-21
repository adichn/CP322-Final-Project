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


def create_detailed_visualizations(df):

    # Set figure size for better readability

    plt.figure(figsize=(20, 15))
    
    # 1. Grade Progression (G1 -> G2 -> G3)

    plt.subplot(3, 3, 1)
    plt.plot([df['G1'].mean(), df['G2'].mean(), df['G3'].mean()], marker='o')
    plt.title('Grade Progression (G1 to G3)')
    plt.xlabel('Period')
    plt.ylabel('Average Grade')
    plt.xticks([0, 1, 2], ['G1', 'G2', 'G3'])
    
    # 2. Weekend Alcohol Consumption vs Grades

    plt.subplot(3, 3, 2)
    walc_means = []
    walc_levels = sorted(df['Walc'].unique())
    for w in walc_levels:
        mean_grade = df[df['Walc'] == w]['G3'].mean()
        walc_means.append(mean_grade)
    plt.bar(walc_levels, walc_means)
    plt.title('Weekend Alcohol Consumption vs Final Grade')
    plt.xlabel('Weekend Alcohol Consumption (1-5)')
    plt.ylabel('Average Final Grade')
    
    # 3. Free Time vs Grades

    plt.subplot(3, 3, 3)
    freetime_means = []
    freetime_levels = sorted(df['freetime'].unique())
    for f in freetime_levels:
        mean_grade = df[df['freetime'] == f]['G3'].mean()
        freetime_means.append(mean_grade)
    plt.bar(freetime_levels, freetime_means)
    plt.title('Free Time vs Final Grade')
    plt.xlabel('Free Time (1-5)')
    plt.ylabel('Average Final Grade')
    
    # 4. Box Plot of Grades by Sex

    plt.subplot(3, 3, 4)
    male_grades = df[df['sex'] == 'M']['G3']
    female_grades = df[df['sex'] == 'F']['G3']
    plt.boxplot([male_grades, female_grades], labels=['Male', 'Female'])
    plt.title('Grade Distribution by Gender')
    plt.ylabel('Final Grade')
    
    # 5. Study Time Distribution

    plt.subplot(3, 3, 5)
    study_counts = df['studytime'].value_counts().sort_index()
    plt.pie(study_counts.values, labels=[f'Level {i}' for i in study_counts.index], 
            autopct='%1.1f%%')
    plt.title('Distribution of Study Time Levels')
    
    # 6. Internet Access vs Grades

    plt.subplot(3, 3, 6)
    internet_means = [df[df['internet'] == 'yes']['G3'].mean(),
                     df[df['internet'] == 'no']['G3'].mean()]
    plt.bar(['Has Internet', 'No Internet'], internet_means)
    plt.title('Internet Access vs Final Grade')
    plt.ylabel('Average Final Grade')
    
    # 7. Absences vs Grades Scatter Plot

    plt.subplot(3, 3, 7)
    plt.scatter(df['absences'], df['G3'], alpha=0.5)
    plt.title('Absences vs Final Grade')
    plt.xlabel('Number of Absences')
    plt.ylabel('Final Grade')
    
    # 8. Father's Education vs Grades

    plt.subplot(3, 3, 8)
    fedu_means = []
    fedu_levels = sorted(df['Fedu'].unique())
    for edu in fedu_levels:
        mean_grade = df[df['Fedu'] == edu]['G3'].mean()
        fedu_means.append(mean_grade)
    plt.bar(fedu_levels, fedu_means)
    plt.title("Father's Education vs Final Grade")
    plt.xlabel("Father's Education Level")
    plt.ylabel('Average Final Grade')
    
    # 9. Health vs Grades

    plt.subplot(3, 3, 9)
    health_means = []
    health_levels = sorted(df['health'].unique())
    for h in health_levels:
        mean_grade = df[df['health'] == h]['G3'].mean()
        health_means.append(mean_grade)
    plt.bar(health_levels, health_means)
    plt.title('Health Status vs Final Grade')
    plt.xlabel('Health Status (1-5)')
    plt.ylabel('Average Final Grade')
    
    plt.tight_layout()
    plt.show()
    
    # Additional plots that don't fit in the 3x3 grid, creating separate figures for these
    
    # 10. Grade Distribution by Address (Urban/Rural)

    plt.figure(figsize=(10, 5))
    urban_grades = df[df['address'] == 'U']['G3']
    rural_grades = df[df['address'] == 'R']['G3']
    plt.boxplot([urban_grades, rural_grades], labels=['Urban', 'Rural'])
    plt.title('Grade Distribution by Address Type')
    plt.ylabel('Final Grade')
    plt.show()
    
    # 11. Correlation Heatmap for Numerical Variables

    plt.figure(figsize=(12, 8))
    numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                     'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                     'absences', 'G1', 'G2', 'G3']
    correlation_matrix = df[numerical_cols].corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(numerical_cols)), numerical_cols, rotation=45, ha='right')
    plt.yticks(range(len(numerical_cols)), numerical_cols)
    plt.title('Correlation Heatmap of Numerical Variables')
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

    # Create detailed visualizations 
    
    create_detailed_visualizations(df)
    
    # Prepare the data for modeling

    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print("\nData preparation completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
except Exception as e:

    print(f"An error occurred: {str(e)}")