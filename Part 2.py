import ucimlrepo as uc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y
                                                            , random_state=42
                                                            )
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        balanced = is_balanced(y_train_series, y_test_series, threshold)

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def perform_cross_validation(X, y, best_params, n_folds=5):
    """
    Perform n-fold cross-validation using the best parameters found
    """
    # Create KNN classifier with best parameters
    best_knn = KNeighborsClassifier(**best_params)

    # Create KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(best_knn, X, y, cv=kf, scoring='recall')

    print(f"\n{n_folds}-Fold Cross-validation Results:")
    print(f"Individual fold scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    return cv_scores


def plot_cv_results(cv_scores, title):
    """
    Plot cross-validation results
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-')
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean CV Score: {cv_scores.mean():.3f}')
    plt.fill_between(range(1, len(cv_scores) + 1),
                     cv_scores.mean() - cv_scores.std(),
                     cv_scores.mean() + cv_scores.std(),
                     alpha=0.2, color='r')
    plt.title(f'{title} Cross-validation Results')
    plt.xlabel('Fold')
    plt.ylabel('Recall Score')
    plt.legend()
    plt.grid(True)
    plt.show()


def getResults(feature_set, goal, title="", n_folds=5):
    X_train, X_test, y_train, y_test = preprocess_and_split(feature_set, goal)

    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    param_grid = {
        'n_neighbors': list(range(1, 21, 2)),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance'],
        'p': [3]
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)

    # Plot GridSearchCV results
    results = grid_search.cv_results_
    plt.figure(figsize=(15, 10))

    for weight in param_grid['weights']:
        plt.subplot(2, 1, param_grid['weights'].index(weight) + 1)
        for metric in param_grid['metric']:
            mask = (results['param_weights'] == weight) & (results['param_metric'] == metric)
            if metric == 'minkowski':
                mask = mask & (results['param_p'] == 3)
            plt.plot(param_grid['n_neighbors'],
                     results['mean_test_score'][mask],
                     label=f"{metric}",
                     linestyle=['-', '-.', '--'][param_grid['metric'].index(metric)],
                     marker=['o', 's', '^'][param_grid['metric'].index(metric)],
                     markersize=4)

        plt.xlim([0, 20])
        plt.xticks(np.arange(1, 20, step=2))
        plt.title(f'{title} recall with {weight} weighting')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Mean Test Score')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Get and print best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Perform N-fold cross-validation with best parameters
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_set)
    cv_scores = perform_cross_validation(X_scaled, goal, best_params, n_folds)
    plot_cv_results(cv_scores, title)

    # Predict using the best model
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)

    # Convert predictions back to original labels
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)

    # Print classification report and confusion matrix
    print("\nClassification Report:\n", classification_report(y_test_original, y_pred_original))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_original, y_pred_original))


# Run analysis for each feature set
print("FEATURE SET 1\n")
getResults(feature_set1, encoded_target, title="Feature Set 1", n_folds=5)
print("\nFEATURE SET 2\n")
getResults(feature_set2, encoded_target, title="Feature Set 2", n_folds=5)
print("\nFEATURE SET 3\n")
getResults(feature_set3, encoded_target, title="Feature Set 3", n_folds=5)