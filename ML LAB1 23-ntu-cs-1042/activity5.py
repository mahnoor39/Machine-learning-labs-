#23-NTU-CS-1042
#mahnoor shahid

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Given dataset
data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}
df = pd.DataFrame(data)
print("Original:\n", df, "\n")

# 1. Handle missing values properly
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
print("After handling missing values:\n", df, "\n")

# 2. Encode Department column
df["Department_Label"] = LabelEncoder().fit_transform(df["Department"])
df = pd.get_dummies(df, columns=["Department"], prefix="Dept")
dept_cols = [c for c in df.columns if c.startswith("Dept_")]
df[dept_cols] = df[dept_cols].astype(int)
print("Encoded:\n", df, "\n")

# 3. Boxplot to detect outliers in Salary
plt.figure(figsize=(5, 6))
plt.boxplot(df["Salary"], vert=True)
plt.title("Boxplot of Salary")
plt.ylabel("Salary")
plt.tight_layout()
plt.show()

# 4. Outlier check using IQR
Q1, Q3 = df["Salary"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
print(f"IQR bounds: lower={lower}, upper={upper}")

zara_salary = df.loc[df["Name"] == "Zara", "Salary"].values[0]
if zara_salary < lower or zara_salary > upper:
    print(f"\nZara's salary ({zara_salary}) IS an outlier — it falls far outside "
          f"the normal IQR range ({lower:.0f} to {upper:.0f}), well above what the "
          f"rest of the team earns.")
else:
    print(f"\nZara's salary ({zara_salary}) is NOT an outlier.")