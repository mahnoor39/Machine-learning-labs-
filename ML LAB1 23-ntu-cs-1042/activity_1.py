import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#23-ntu-cs-1042
#mahnoor shahid 
# 1. Dataset (with missing values to clean)
df = pd.DataFrame({
    "Province": ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan"],
    "Population (millions)": [127.7, 55.7, np.nan, 15.0],
    "Literacy Rate (%)": [62.0, np.nan, 53.0, 46.0],
    "Region": ["North", "South", "West", "East"]
})
print("Original:\n", df, "\n")

# 2. Handle missing values
df = df.dropna(subset=["Population (millions)"]).reset_index(drop=True)
df["Literacy Rate (%)"] = df["Literacy Rate (%)"].fillna(df["Literacy Rate (%)"].median())
print("Cleaned:\n", df, "\n")

# 3. Encode Region
df["Region_Label"] = LabelEncoder().fit_transform(df["Region"])
onehot = pd.get_dummies(df["Region"], prefix="Region").astype(int)
df = pd.concat([df, onehot], axis=1)
print("Encoded:\n", df, "\n")

# 4. Scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(df["Population (millions)"], df["Literacy Rate (%)"], s=100, color="royalblue", edgecolor="black")
for _, row in df.iterrows():
    plt.annotate(row["Province"], (row["Population (millions)"], row["Literacy Rate (%)"]), xytext=(8, 6), textcoords="offset points")
plt.title("Population vs Literacy Rate")
plt.xlabel("Population (millions)")
plt.ylabel("Literacy Rate (%)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Outlier detection (IQR method)
lit = df["Literacy Rate (%)"]
Q1, Q3 = lit.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(lit < lower) | (lit > upper)]

print("Outliers found:\n", outliers[["Province", "Literacy Rate (%)"]] if not outliers.empty else "None")