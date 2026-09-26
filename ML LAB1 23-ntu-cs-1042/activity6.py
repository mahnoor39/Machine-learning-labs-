#23-NTU-CS-1042 
#mahnoor shahid

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE

# 1. Explore the dataset
data = load_breast_cancer()

print("Keys:", data.keys())
print("Target names:", data.target_names)
print("Number of samples:", data.data.shape[0])
print("Number of features:", data.data.shape[1])
print("Feature names:", data.feature_names)
print("Shape of data.data:", data.data.shape)
print("Shape of data.target:", data.target.shape)

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# 2. Check for missing values
print("\nMissing values before:\n", df.isnull().sum().sum())

# Artificially introduce 5 missing values in 'mean radius'
np.random.seed(42)
missing_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[missing_idx, "mean radius"] = np.nan
print("\nMissing values after introducing NaNs:\n", df["mean radius"].isnull().sum())

# Handle missing values: use median (robust to outliers, better than mean for
# a feature like this that can be skewed by extreme tumor sizes)
df["mean radius"] = df["mean radius"].fillna(df["mean radius"].median())
print("Missing values after filling:", df["mean radius"].isnull().sum())

# 3. Class distribution (malignant vs benign)
print("\nClass distribution before balancing:")
print(df["target"].value_counts().rename(index={0: "malignant", 1: "benign"}))

counts = df["target"].value_counts()
ratio = counts.max() / counts.min()
print(f"\nImbalance ratio: {ratio:.2f}")
if ratio > 1.5:
    print("Dataset is imbalanced -> applying SMOTE")

    X = df.drop(columns=["target"])
    y = df["target"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("\nClass distribution after SMOTE:")
    print(y_res.value_counts().rename(index={0: "malignant", 1: "benign"}))
else:
    print("Dataset is roughly balanced -> no resampling needed")
