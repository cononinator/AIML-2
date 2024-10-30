import ucimlrepo as uc
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

# Load the dataset
bcwd = uc.fetch_ucirepo(id=17)

feature = bcwd.data.features
print(feature.describe())
target = bcwd.data.targets
# Prepare feature sets
data = feature.iloc[:, :10]  # First 10 features (mean values)

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
data_min_max_scaled = min_max_scaler.fit_transform(data)

# Z-score Normalization (Standardization)
standard_scaler = StandardScaler()
data_standard_scaled = standard_scaler.fit_transform(data)

# MaxAbsScaler
max_abs_scaler = MaxAbsScaler()
data_max_abs_scaled = max_abs_scaler.fit_transform(data)

# RobustScaler
robust_scaler = RobustScaler()
data_robust_scaled = robust_scaler.fit_transform(data)

# Normalizer
normalizer = Normalizer()
data_normalized = normalizer.fit_transform(data)

#%%
# Print results
print("Min-Max Scaled Data:\n", data_min_max_scaled)
print("Standard Scaled Data:\n", data_standard_scaled)
print("MaxAbs Scaled Data:\n", data_max_abs_scaled)
print("Robust Scaled Data:\n", data_robust_scaled)
print("Normalized Data:\n", data_normalized)