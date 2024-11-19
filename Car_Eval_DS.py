import ssl
from ucimlrepo import fetch_ucirepo

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Fetch the dataset
car_evaluation = fetch_ucirepo(id=19)

# Features and targets
X = car_evaluation.data.features
y = car_evaluation.data.targets

# Print metadata
print("Metadata:\n", car_evaluation.metadata)
print("Variables:\n", car_evaluation.variables)