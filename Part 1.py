import ucimlrepo as uc
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
bcwd = uc.fetch_ucirepo(id=17)

feature = bcwd.data.features
target = bcwd.data.targets


print(f"Feature shape: {feature.shape}")
print(f"Target shape: {target.shape}")

# Check for missing values
print("\nMissing values:")
print(feature.isnull().sum())

# Basic statistics of features
print("\nFeature statistics:")
print(feature.describe())

# Class distribution
print("\nClass distribution:")
print(target['Diagnosis'].value_counts(normalize=True))

# Visualize distribution of a few features
plt.figure(figsize=(15, 5))
for i, col in enumerate(['radius1', 'texture1', 'perimeter1']):
    plt.subplot(1, 3, i+1)
    feature[col].hist()
    plt.title(col)
plt.tight_layout()
plt.show()

# Prepare feature sets
feature_set1 = feature.iloc[:, :10]  # First 10 features (mean values)
feature_set2 = feature.iloc[:, :20]  # First 20 features (mean and standard error)
feature_set3 = feature  # All 30 features

# Function to check if class distribution is balanced within a threshold
def is_balanced(y_train, y_test, threshold=0.1):
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    for cls in train_dist.index:
        if abs(train_dist[cls] - test_dist[cls]) > threshold:
            return False
    return True


# Function to preprocess and split data with balance check
def preprocess_and_split(X, y, test_size=0.3, normalize=True, threshold=0.1):
    balanced = False
    while not balanced:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        balanced = is_balanced(y_train, y_test, threshold)

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# Example usage:
X_train, X_test, y_train, y_test = preprocess_and_split(feature_set1, target['Diagnosis'])

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

