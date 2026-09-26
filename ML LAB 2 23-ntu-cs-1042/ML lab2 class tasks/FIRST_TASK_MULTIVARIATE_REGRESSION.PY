#23-ntu-cs-1042
#mahnoor shahid 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Admission_Predict.csv")
y = df["Chance of Admit "]

simple_lr = LinearRegression()
simple_lr.fit(df[['GRE Score']], y)

plt.scatter(df['GRE Score'], y, color='green', label='Actual Data', alpha=0.5)
plt.plot(df['GRE Score'], simple_lr.predict(df[['GRE Score']]), color='red', linewidth=3, label='Regression Line')

plt.xlabel('GRE Score')
plt.ylabel('Admission chance')
plt.title('GRE Score vs Admission chance')
plt.legend()
plt.show()