#23-NTU-CS-1042
#mahnoor shahid

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load all three files into DataFrames
marks_df = pd.read_csv("students.csv")
attendance_df = pd.read_json("attendance.json")
bonus_df = pd.read_excel("extra.xlsx")

# 2. Merge into a single DataFrame: Name, Marks, Attendance, Bonus
df = marks_df.merge(attendance_df, on="Name").merge(bonus_df, on="Name")
print(df.head())

# 3. Scatter plot: Marks vs Attendance, highlight students with attendance < 70%
low_attendance = df[df["Attendance"] < 70]
normal = df[df["Attendance"] >= 70]

plt.figure(figsize=(8, 6))
plt.scatter(normal["Attendance"], normal["Marks"], color="royalblue", label="Attendance >= 70%")
plt.scatter(low_attendance["Attendance"], low_attendance["Marks"], color="red", label="Attendance < 70%")
plt.title("Marks vs Attendance")
plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.legend()
plt.tight_layout()
plt.show()

# 4. One-hot encoding on Name
df_encoded = pd.get_dummies(df, columns=["Name"]).astype(int, errors="ignore")
print(df_encoded.head())
