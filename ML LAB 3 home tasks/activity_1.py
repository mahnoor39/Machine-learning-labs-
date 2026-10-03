# ===========================================================
# Mahnoor Shahid 
# 23-ntu-cs-1042
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)


data = pd.read_csv(r'D:\7th semester all subjects\ML LABS\ML LAB 3 23-ntu-cs-1042\lab 3 home tasks\Data Set\Data Set\Telco Customer Churn.csv')


# Select required features + target
data = data[['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'Churn']]


# Encoding categorical variables
data = pd.get_dummies(data, columns=['Contract', 'InternetService'], drop_first=True, dtype=int)
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})


X = data.drop('Churn', axis=1).values
y = data['Churn'].values  # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Update weights and bias by calculating Gradient
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        return [1 if i > 0.5 else 0 for i in self.predict_proba(X)]

model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1-score:", f1_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))


# Plot Confusion Matrix & ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=["No Churn", "Churn"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix")

fpr, tpr, _ = roc_curve(y_test, probabilities)
axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_score(y_test, probabilities):.3f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random guess")
axes[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()