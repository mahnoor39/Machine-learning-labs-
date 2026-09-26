#23-ntu-cs-1042
#Mahnoor Shahid
# # Activity 3: 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"D:\7th semester all subjects\ML LABS\ML LAB 2\LAB 2 ACTIVITIES\Data Set\Data Set\Car Price Prediction.csv")

X = df[['horsepower', 'enginesize', 'curbweight']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

print("Linear Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, lin_pred)))
print("R2 Score:", r2_score(y_test, lin_pred))

for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_pred = poly_model.predict(X_test_poly)

    rmse = np.sqrt(mean_squared_error(y_test, poly_pred))
    r2 = r2_score(y_test, poly_pred)
    print(f"\nPolynomial Regression (degree={degree}):")
    print("RMSE:", rmse)
    print("R2 Score:", r2)

X_hp = df[['horsepower']].values
y_price = df['price'].values

plt.scatter(X_hp, y_price, color='blue', label='Actual Data', alpha=0.5)

for degree, color in zip([2, 3, 4], ['red', 'green', 'orange']):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_hp)
    model = LinearRegression()
    model.fit(X_poly, y_price)

    X_line = np.linspace(X_hp.min(), X_hp.max(), 200).reshape(-1, 1)
    y_line = model.predict(poly.transform(X_line))
    plt.plot(X_line, y_line, color=color, label=f'Degree {degree}')

plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Polynomial Regression Fit (Horsepower vs Price)')
plt.legend()
plt.show()