import ucimlrepo as uc
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
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
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
        'weights': ['uniform', 'distance'],
        'p': [3]
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return recall_score(y_test, y_pred), best_knn.get_params(), conf_matrix


def plot_confusion_matrices(confusion_matrices):
    # Plot separate confusion matrix for each feature set
    for feature_name, classifier_matrices in confusion_matrices.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Confusion Matrices for {feature_name}', fontsize=14, y=1.02)

        # Flatten axes for easier iteration
        axes = axes.flatten()

        for idx, (classifier_name, conf_matrix) in enumerate(classifier_matrices.items()):
            # Calculate percentages for annotations
            conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

            # Create annotation text with both count and percentage
            annot = np.array([[f'{count}\n({percent:.1f}%)'
                               for count, percent in zip(row_counts, row_percents)]
                              for row_counts, row_percents in zip(conf_matrix, conf_matrix_percent)])

            sns.heatmap(conf_matrix, annot=annot, fmt='', cmap='Blues', ax=axes[idx],
                        cbar=True, square=True)
            axes[idx].set_title(f'{classifier_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

            # Add class labels
            axes[idx].set_xticklabels(['Benign', 'Malignant'])
            axes[idx].set_yticklabels(['Benign', 'Malignant'])

        plt.tight_layout()
        plt.show()


def plot_decision_boundaries(X, y, classifiers, feature_set_name, normalize=True):
    """
    Plot decision boundaries for multiple classifiers using PCA for dimensionality reduction.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    classifiers : dict
        Dictionary of classifier names and fitted classifier objects
    feature_set_name : str
        Name of the feature set being visualized
    normalize : bool, default=True
        Whether to normalize the features before PCA
    """
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create meshgrid for decision boundary
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Set up the subplot layout
    n_classifiers = len(classifiers)
    n_cols = 2
    n_rows = (n_classifiers + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = np.array(axes).reshape(-1)  # Ensure axes is always a 1D array

    # Plot for each classifier
    for idx, (name, clf) in enumerate(classifiers.items()):
        ax = axes[idx]

        # Try to use predict_proba if available, otherwise use predict
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        try:
            # For probabilistic classifiers
            Z = clf.predict_proba(pca.inverse_transform(mesh_points))[:, 1]
        except AttributeError:
            # For non-probabilistic classifiers
            Z = clf.predict(pca.inverse_transform(mesh_points))

        # Reshape the predictions
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and data points
        contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), alpha=0.4, cmap='RdYlBu')
        scatter1 = ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='blue',
                              marker='o', label='Benign', alpha=0.6)
        scatter2 = ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red',
                              marker='^', label='Malignant', alpha=0.6)

        ax.set_title(f'{name} Decision Boundary')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.legend()

        # Add colorbar
        plt.colorbar(contour, ax=ax)

    # Remove empty subplots if any
    if n_classifiers < len(axes):
        for idx in range(n_classifiers, len(axes)):
            fig.delaxes(axes[idx])

    plt.suptitle(f'Decision Boundaries for {feature_set_name}', y=1.02)
    plt.tight_layout()
    plt.show()


# Example usage in your existing code:
def evaluate_and_plot_boundaries(feature_sets, encoded_target):
    for feature_name, features in feature_sets.items():
        X_train, X_test, y_train, y_test = preprocess_and_split(features, encoded_target)

        # Initialize and fit classifiers
        classifiers = {
            'KNN': KNeighborsClassifier(**best_params[feature_name]['KNN']),
            'Gaussian NB': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis()
        }

        # Fit all classifiers
        fitted_classifiers = {}
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            fitted_classifiers[name] = clf

        # Plot decision boundaries
        plot_decision_boundaries(X_train, y_train, fitted_classifiers, feature_name)

def evaluate_all_classifiers(feature_sets):
    results = {}
    best_params = {}
    confusion_matrices = {}

    classifiers = {
        'KNN': None,  # Will be set by GridSearchCV
        'Gaussian NB': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis()
    }



    for feature_name, features in feature_sets.items():
        results[feature_name] = {}
        best_params[feature_name] = {}
        confusion_matrices[feature_name] = {}

        X_train, X_test, y_train, y_test = preprocess_and_split(features, encoded_target)

        # Get best KNN first
        knn_score, knn_params, knn_conf_matrix = get_best_knn(X_train, X_test, y_train, y_test)
        results[feature_name]['KNN'] = knn_score
        best_params[feature_name]['KNN'] = knn_params
        confusion_matrices[feature_name]['KNN'] = knn_conf_matrix

        # Evaluate other classifiers
        for name, clf in list(classifiers.items())[1:]:  # Skip KNN as it's already done
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results[feature_name][name] = recall_score(y_test, y_pred)
            confusion_matrices[feature_name][name] = confusion_matrix(y_test, y_pred)


    return results, best_params, confusion_matrices


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
results, best_params, confusion_matrices = evaluate_all_classifiers(feature_sets)

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
plot_confusion_matrices(confusion_matrices)

# After your existing evaluation
print("Plotting decision boundaries...")
evaluate_and_plot_boundaries(feature_sets, encoded_target)