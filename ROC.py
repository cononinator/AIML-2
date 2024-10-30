import ucimlrepo as uc
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
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
        X, y, test_size=test_size, stratify=y, random_state=42
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
    y_prob = best_knn.predict_proba(X_test)[:, 1]
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate ROC curve and AUC for KNN
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    return recall_score(y_test, y_pred), best_knn.get_params(), conf_matrix, (fpr, tpr, auc_score)


def evaluate_all_classifiers(feature_sets):
    results = {}
    best_params = {}
    confusion_matrices = {}
    roc_curves = {}
    auc_scores = {}  # New dictionary to store AUC scores

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
        roc_curves[feature_name] = {}
        auc_scores[feature_name] = {}  # Initialize AUC scores for this feature set

        X_train, X_test, y_train, y_test = preprocess_and_split(features, encoded_target)

        # Get best KNN first
        knn_score, knn_params, knn_conf_matrix, knn_roc = get_best_knn(X_train, X_test, y_train, y_test)
        results[feature_name]['KNN'] = knn_score
        best_params[feature_name]['KNN'] = knn_params
        confusion_matrices[feature_name]['KNN'] = knn_conf_matrix
        roc_curves[feature_name]['KNN'] = knn_roc
        auc_scores[feature_name]['KNN'] = knn_roc[2]

        # Evaluate other classifiers
        for name, clf in list(classifiers.items())[1:]:  # Skip KNN as it's already done
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            results[feature_name][name] = recall_score(y_test, y_pred)
            confusion_matrices[feature_name][name] = confusion_matrix(y_test, y_pred)

            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)
            roc_curves[feature_name][name] = (fpr, tpr, auc_score)
            auc_scores[feature_name][name] = auc_score

    return results, best_params, confusion_matrices, roc_curves, auc_scores


def plot_roc_curves(roc_curves):
    n_feature_sets = len(roc_curves)
    fig, axes = plt.subplots(1, n_feature_sets, figsize=(6 * n_feature_sets, 5))

    if n_feature_sets == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, 4))

    for idx, (feature_name, classifiers) in enumerate(roc_curves.items()):
        ax = axes[idx]

        for (classifier_name, (fpr, tpr, auc_score)), color in zip(classifiers.items(), colors):
            ax.plot(fpr, tpr, color=color, label=f"{classifier_name} (AUC = {auc_score:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {feature_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_auc_comparison(auc_scores):
    # Convert AUC scores to DataFrame for easier plotting
    auc_df = pd.DataFrame(auc_scores).melt(var_name='Feature Set', value_name='AUC', ignore_index=False)
    auc_df = auc_df.reset_index().rename(columns={'index': 'Classifier'})

    plt.figure(figsize=(12, 6))
    sns.barplot(data=auc_df, x='Classifier', y='AUC', hue='Feature Set')
    plt.title('AUC Scores Comparison Across Feature Sets and Classifiers')
    plt.xticks(rotation=45)
    plt.ylim((0.9, 1))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# Run the analysis
print("Evaluating classifiers...")
results, best_params, confusion_matrices, roc_curves, auc_scores = evaluate_all_classifiers(feature_sets)

# Print detailed results
print("\nDetailed Results:")
print("-" * 50)
for feature_set, scores in results.items():
    print(f"\nFeature Set: {feature_set}")
    print("Classifier Performance:")
    print("{:<12} {:<10} {:<10}".format("Classifier", "Recall", "AUC"))
    print("-" * 32)
    for classifier, recall in scores.items():
        auc = auc_scores[feature_set][classifier]
        print("{:<12} {:<10.3f} {:<10.3f}".format(classifier, recall, auc))
    if feature_set in best_params:
        print(f"\nBest KNN parameters for {feature_set}:")
        print(best_params[feature_set]['KNN'])

# Plot the results
plot_roc_curves(roc_curves)
plot_auc_comparison(auc_scores)