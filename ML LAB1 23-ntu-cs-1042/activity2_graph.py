import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

# 1. Generate random data for 100 students
n = 100
df = pd.DataFrame({
    "ID": range(1, n + 1),
    "Math": np.random.randint(30, 100, n).astype(float),
    "Science": np.random.randint(30, 100, n).astype(float),
    "English": np.random.randint(30, 100, n).astype(float),
    "Grade": np.random.choice(["A", "B", "C"], n)
})

# insert some missing marks
for col in ["Math", "Science", "English"]:
    df.loc[df.sample(5, random_state=1).index, col] = np.nan

# 2. Handle missing marks (fill with mean)
df[["Math", "Science", "English"]] = df[["Math", "Science", "English"]].fillna(
    df[["Math", "Science", "English"]].mean()
)

# 3. Encode Grade column
df["Grade_Label"] = LabelEncoder().fit_transform(df["Grade"])

# 4. Histogram of Math scores
plt.figure(figsize=(7, 5))
plt.hist(df["Math"], bins=10, color="royalblue", edgecolor="black")
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 5. Boxplot to identify unusually high/low total scores
df["Total"] = df["Math"] + df["Science"] + df["English"]

plt.figure(figsize=(5, 6))
plt.boxplot(df["Total"], vert=True)
plt.title("Boxplot of Total Scores")
plt.ylabel("Total Score")
plt.tight_layout()
plt.show()

Q1, Q3 = df["Total"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df["Total"] < lower) | (df["Total"] > upper)]

print("Students with unusually high/low total score:\n",
      outliers[["ID", "Total"]] if not outliers.empty else "None")

# 6. Summarize findings
summary = df.groupby("Grade")["Total"].mean().sort_values(ascending=False)
print("\nAverage total score by grade:\n", summary)
print(f"\nBest performing grade overall: {summary.idxmax()}")
