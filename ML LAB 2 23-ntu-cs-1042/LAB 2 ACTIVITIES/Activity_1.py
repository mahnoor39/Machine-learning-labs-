#23-ntu-cs-1042
#Mahnoor Shahid 
# Activity 1:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"D:\7th semester all subjects\ML LABS\ML LAB 2\LAB 2 ACTIVITIES\Data Set\Data Set\Medical Cost Personal Datasets.csv")
df.head()

le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])          # male=1, female=0
df['smoker'] = le_smoker.fit_transform(df['smoker'])  # yes=1, no=0
df['region'] = le_region.fit_transform(df['region'])  # 0,1,2,3

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Perfect Prediction Line')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Predicted vs Actual Insurance Costs')
plt.legend()
plt.show()