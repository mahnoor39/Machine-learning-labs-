#23-ntu-cs-1042
#Mahnoor Shahid
# Activity 2: 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix,
                              ConfusionMatrixDisplay, RocCurveDisplay)

df = pd.read_csv(r"D:\7th semester all subjects\ML LABS\ML LAB 2\LAB 2 ACTIVITIES\Data Set\Data Set\Telco Customer Churn.csv")

df = df[['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'Churn']]

le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['Churn'] = le.fit_transform(df['Churn'])  # Yes=1, No=0

X = df[['tenure', 'MonthlyCharges', 'Contract', 'InternetService']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()