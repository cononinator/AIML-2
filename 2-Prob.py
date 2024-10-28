import ucimlrepo as uc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
bcwd = uc.fetch_ucirepo(id=17)

feature = bcwd.data.features
target = bcwd.data.targets

# Encode the target variable
le = LabelEncoder()
encoded_target = le.fit_transform(target['Diagnosis'])

# Prepare feature sets
feature_set1 = feature.iloc[:, :10]  # First 10 features (mean values)
feature_set2 = feature.iloc[:, :20]  # First 20 features (mean and standard error)
feature_set3 = feature  # All 30 features


def is_balanced(y_train, y_test, threshold=0.1):
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    for cls in train_dist.index:
        if abs(train_dist[cls] - test_dist[cls]) > threshold:
            return False
    return True


def preprocess_and_split(X, y, test_size=0.3, normalize=True, threshold=0.1):
    balanced = False
    while not balanced:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        balanced = is_balanced(y_train_series, y_test_series, threshold)

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate_classifier(clf, X_train, X_test, y_train, y_test, le, clf_name):
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Convert numerical predictions back to original labels
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)

    # Print results
    print(f"\n{clf_name} Results:")
    print("\nClassification Report:\n", classification_report(y_test_original, y_pred_original))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_original, y_pred_original))

    return clf.score(X_test, y_test)


def compare_classifiers(feature_set, goal, title=""):
    print(f"\n{title}")
    print("-" * 50)

    X_train, X_test, y_train, y_test = preprocess_and_split(feature_set, goal)

    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    # Initialize classifiers
    classifiers = {
        'Gaussian Naive Bayes': GaussianNB(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis()
    }

    # Store accuracy scores for plotting
    scores = {}

    # Evaluate each classifier
    for name, clf in classifiers.items():
        score = evaluate_classifier(clf, X_train, X_test, y_train, y_test, le, name)
        scores[name] = score

    # Plot results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scores.keys(), scores.values())
    plt.title(f'{title} - Classifier Comparison')
    plt.ylabel('Accuracy Score')
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# Run comparison for each feature set
print("Comparing classifiers across different feature sets...")
compare_classifiers(feature_set1, encoded_target, title="Feature Set 1 (Mean Values)")
compare_classifiers(feature_set2, encoded_target, title="Feature Set 2 (Mean + Standard Error)")
compare_classifiers(feature_set3, encoded_target, title="Feature Set 3 (All Features)")