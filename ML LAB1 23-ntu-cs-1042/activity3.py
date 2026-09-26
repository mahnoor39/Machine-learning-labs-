#23-ntu-cs-1042
#mahnoor shahid
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the online dataset
url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"
df = pd.read_csv(url)

# 2. Extract only Pakistan's data
pk = df[df["Country/Region"] == "Pakistan"].copy()
pk["Date"] = pd.to_datetime(pk["Date"])
pk = pk.sort_values("Date").reset_index(drop=True)

# 3. Handle missing values (if any)
print("Missing values:\n", pk.isnull().sum())
pk[["Confirmed", "Recovered", "Deaths"]] = pk[["Confirmed", "Recovered", "Deaths"]].fillna(0)

# 4. Daily new cases (dataset gives cumulative confirmed cases)
pk["Daily_Cases"] = pk["Confirmed"].diff().fillna(0)

# 5. Line chart: cases over time
plt.figure(figsize=(10, 5))
plt.plot(pk["Date"], pk["Confirmed"], color="royalblue")
plt.title("Pakistan - Cumulative Confirmed COVID-19 Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.tight_layout()
plt.show()

# 6. Day with the highest confirmed (daily new) cases
peak_day = pk.loc[pk["Daily_Cases"].idxmax()]
print(f"\nDay with highest daily confirmed cases: {peak_day['Date'].date()} "
      f"({int(peak_day['Daily_Cases'])} cases)")

# 7. Boxplot to detect outliers in daily cases
plt.figure(figsize=(5, 6))
plt.boxplot(pk["Daily_Cases"], vert=True)
plt.title("Boxplot of Daily Confirmed Cases (Pakistan)")
plt.ylabel("Daily Cases")
plt.tight_layout()
plt.show()

Q1, Q3 = pk["Daily_Cases"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = pk[(pk["Daily_Cases"] < lower) | (pk["Daily_Cases"] > upper)]

print(f"\nNumber of outlier days in daily cases: {len(outliers)}")
print(outliers[["Date", "Daily_Cases"]].head())
