# Wisconsin Breast Cancer Detection (WBCD) Dataset Analysis Plan

## 1. Data Exploration and Preprocessing

### 1.1 Dataset Completeness
- Check for missing values in all 30 features
- Report on data completeness

### 1.2 Feature Normalization
- Analyze feature distributions
- Implement and evaluate normalization techniques (e.g., Min-Max scaling, Z-score normalization)

### 1.3 Train-Test Split
- Determine appropriate test set size (e.g., 20-30% of data)
- Ensure balanced class distribution in both sets

## 2. Classifier Evaluation

### 2.1 k-Nearest Neighbors (k-NN)
- Implement k-NN classifier
- Perform hyperparameter tuning:
  - Optimal k value (odd numbers from 1 to 20)
  - Distance metrics (Euclidean, Manhattan, Minkowski)
  - Weighting (uniform, distance-based)

### 2.2 Probabilistic Classifiers
- Implement and evaluate:
  - Naive Bayes (Gaussian NB)
  - Linear Discriminant Analysis (LDA)
  - Quadratic Discriminant Analysis (QDA)

## 3. Performance Metrics
- Accuracy
- Precision, Recall, F1-score
- ROC curve and AUC
- Confusion matrix

## 4. Experimental Setup

### 4.1 Feature Sets
- Set 1: First 10 features (mean values)
- Set 2: First 20 features (mean and standard error)
- Set 3: All 30 features (mean, standard error, and maximum values)

### 4.2 Cross-validation
- Implement k-fold cross-validation (e.g., 5-fold or 10-fold)

## 5. Results and Analysis
- Compare classifier performance across feature sets
- Analyze impact of feature normalization
- Discuss optimal hyperparameters for k-NN
- Compare k-NN and probabilistic classifier performance
- Interpret results in the context of breast cancer diagnosis

## 6. Conclusion
- Summarize key findings
- Recommend best-performing model and configuration
- Discuss potential improvements and future work
