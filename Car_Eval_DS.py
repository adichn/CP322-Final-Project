import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset
automobile = fetch_ucirepo(id=10)

# Data (features & target + combination & target isolation)
X = automobile.data.features
y = automobile.data.targets
data = pd.concat([X, y], axis=1)
target_column = y.columns[0] if not y.empty else None  

# 1. Basic Dataset Overview
def dataset_overview(data):
    print("### Dataset Overview ###")
    print(f"Number of Rows: {data.shape[0]}")
    print(f"Number of Columns: {data.shape[1]}")
    print("\nColumn Names:")
    print(data.columns.tolist())
    print("\nFirst 5 Rows of Data:")
    print(data.head())
    print("\n")

# 2. Generate Data Quality Report
def data_quality_report(data):
    report = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes,
        "Unique Values": data.nunique(),
        "Most Frequent Value": data.mode().iloc[0],
        "Missing Values": data.isnull().sum(),
    })
    print("### Data Quality Report ###")
    print(report)
    print("\n")

# 3. Target Variable Distribution
def target_distribution(data, target_column):
    if target_column and target_column in data.columns:
        print(f"### Target Variable Distribution: {target_column} ###")
        print(data[target_column].value_counts(normalize=True) * 100)
    else:
        print("### Target Variable Distribution ###")
        print("Target column is not explicitly provided or identified.")
    print("\n")

# Execute Analysis
print(automobile.metadata) 
print("\n### Variable Information ###\n", automobile.variables)

# Run the analysis functions
dataset_overview(data)
data_quality_report(data)

# Ensure we handle the target column dynamically
if target_column:
    target_distribution(data, target_column)
else:
    print("No target variable identified.")
