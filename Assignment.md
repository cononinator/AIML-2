## 1. Data Exploration and Preprocessing

### 1.1 Dataset Completeness
Dataset Completeness

```python
# Check for missing values
print("\nMissing values:")
print(feature.isnull().sum())

Missing values:
radius1               0
texture1              0
perimeter1            0
area1                 0
...
symmetry3             0
fractal_dimension3    0
dtype: int64
```

## 1.2 Feature Normalization
Choose a normalization technique and explain why you chose it.
For K-NN, LDA and QDA Z-Score Normilization was chosen as it preserves outliers
These are susceptible to different scale factors between features. Z-Score normalization is a common technique used to address this issue. It standardizes the 
data by subtracting the mean and dividing by the standard deviation of each feature.

For Naive Bayes no normalization was chosen as it is not sensitive to different scale factors between features.
```python
# Basic statistics of features
print("\nFeature statistics:")
print(feature.describe())
```
Area1 has a scale in the 1000s whereas fractal_dimension1 has a scale in the 0.00s. 
This is a problem for K-NN, LDA and QDA as they are susceptible to different scale factors between features. Z-Score normalization is a common technique used to address this issue. It standardizes the data by subtracting the mean and dividing by the standard deviation of each feature.

TODO add Compare unscaled versus scaled algorithm effictiveness

## 1.3 Train-Test Split
Split at 30% test size
How to verify mix of classes in both sets


## 2. Classifier Evaluation
### 2.1 k-Nearest Neighbors (k-NN)
Comparing the different distance metrics and weighting schemes
- Eucledian
- Manhattan
- Minkowski

- Uniform
- Distance-based

Issues with comparison, Euclidean and Minkowski are mathematically equivalent when p=2. P=3 they are different.

````console
FEATURE SET 1


Training set shape: (398, 10)
Test set shape: (171, 10)
E:\Documents\Engineering\AI&ML\BreastCancer\.venv\Lib\site-packages\numpy\ma\core.py:2881: RuntimeWarning: invalid value encountered in cast
  _data = np.array(data, dtype=dtype, copy=copy,
Best parameters found: {'metric': 'manhattan', 'n_neighbors': 7, 'p': 3, 'weights': 'uniform'}

Classification Report:
               precision    recall  f1-score   support

           B       0.95      0.95      0.95       107
           M       0.92      0.92      0.92        64

    accuracy                           0.94       171
   macro avg       0.94      0.94      0.94       171
weighted avg       0.94      0.94      0.94       171


Confusion Matrix:
[[102   5]
 [  5  59]]
FEATURE SET 2


Training set shape: (398, 20)
Test set shape: (171, 20)
Best parameters found: {'metric': 'manhattan', 'n_neighbors': 11, 'p': 3, 'weights': 'distance'}

Classification Report:
               precision    recall  f1-score   support

           B       0.94      0.99      0.96       107
           M       0.98      0.89      0.93        64

    accuracy                           0.95       171
   macro avg       0.96      0.94      0.95       171
weighted avg       0.95      0.95      0.95       171


Confusion Matrix:
[[106   1]
 [  7  57]]
FEATURE SET 3


Training set shape: (398, 30)
Test set shape: (171, 30)
Best parameters found: {'metric': 'manhattan', 'n_neighbors': 5, 'p': 3, 'weights': 'uniform'}

Classification Report:
               precision    recall  f1-score   support

           B       0.96      1.00      0.98       107
           M       1.00      0.92      0.96        64

    accuracy                           0.97       171
   macro avg       0.98      0.96      0.97       171
weighted avg       0.97      0.97      0.97       171


Confusion Matrix:
[[107   0]
 [  5  59]]

````