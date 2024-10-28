import ucimlrepo as uc
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the dataset
bcwd = uc.fetch_ucirepo(id=17)
feature = bcwd.data.features
target = bcwd.data.targets

# Encode the target variable
le = LabelEncoder()
encoded_target = le.fit_transform(target['Diagnosis'])

# Prepare feature sets
feature_sets = {
    'Mean Values': feature.iloc[:, :10],
    'Mean + Std Error': feature.iloc[:, :20],
    'All Features': feature
}


def preprocess_and_split(X, y, test_size=0.3, normalize=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
        , random_state=42
    )

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def get_best_knn(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_neighbors': list(range(1, 21, 2)),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance'],
        'p': [3]
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)

    return recall_score(y_test, y_pred), best_knn.get_params()


def evaluate_all_classifiers(feature_sets):
    results = {}
    best_params = {}

    classifiers = {
        'KNN': None,  # Will be set by GridSearchCV
        'Gaussian NB': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

    for feature_name, features in feature_sets.items():
        results[feature_name] = {}
        best_params[feature_name] = {}

        X_train, X_test, y_train, y_test = preprocess_and_split(features, encoded_target)

        # Get best KNN first
        knn_score, knn_params = get_best_knn(X_train, X_test, y_train, y_test)
        results[feature_name]['KNN'] = knn_score
        best_params[feature_name]['KNN'] = knn_params

        # Evaluate other classifiers
        for name, clf in list(classifiers.items())[1:]:  # Skip KNN as it's already done
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results[feature_name][name] = recall_score(y_test, y_pred)

    return results, best_params


def plot_comparison(results):
    plt.figure(figsize=(15, 8))

    # Prepare data for plotting
    feature_sets = list(results.keys())
    classifiers = list(results[feature_sets[0]].keys())
    x = np.arange(len(feature_sets))
    width = 0.2
    multiplier = 0

    # Plot bars for each classifier
    for classifier in classifiers:
        scores = [results[feature_set][classifier] for feature_set in feature_sets]
        offset = width * multiplier
        rects = plt.bar(x + offset, scores, width, label=classifier)
        multiplier += 1

    # Customize the plot
    plt.xlabel('Feature Sets')
    plt.ylabel('Recall Score')
    plt.title('Classifier Performance Comparison')
    plt.xticks(x + width * (len(classifiers) - 1) / 2, feature_sets, rotation=45)
    plt.ylim([0.6, 1.05])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def plot_heatmap(results):
    # Convert results to DataFrame for heatmap
    df_results = pd.DataFrame(results).round(3)

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_results, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Classifier Performance Heatmap')
    plt.ylabel('Classifiers')
    plt.xlabel('Feature Sets')
    plt.tight_layout()
    plt.show()


# Run the analysis
print("Evaluating classifiers...")
results, best_params = evaluate_all_classifiers(feature_sets)

# Print detailed results
print("\nDetailed Results:")
print("-" * 50)
for feature_set, scores in results.items():
    print(f"\nFeature Set: {feature_set}")
    print("Classifier Performance (Recall):")
    for classifier, score in scores.items():
        print(f"{classifier}: {score:.3f}")
    if feature_set in best_params:
        print(f"\nBest KNN parameters for {feature_set}:")
        print(best_params[feature_set]['KNN'])

# Plot the results
plot_comparison(results)
plot_heatmap(results)