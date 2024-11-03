import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


np.random.seed(42)


n_samples = 1000


data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'debt_ratio': np.random.uniform(0, 1, n_samples),
    'loan_amount': np.random.normal(15000, 5000, n_samples)
})


data['loan_approved'] = (
    (data['income'] > 40000) & 
    (data['credit_score'] > 600) & 
    (data['debt_ratio'] < 0.4)
).astype(int)
data.to_csv('synthetic_loan_data.csv', index=False)

print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10, 6))
sns.histplot(data['age'], kde=True)
plt.title('Age Distribution')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

X = data.drop('loan_approved', axis=1)
y = data['loan_approved']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



joblib.dump(model, './models/credit_approval_model.joblib')
joblib.dump(scaler, './models/scaler.joblib')

loaded_model=joblib.load('./models/credit_approval_model.joblib')
print(type(loaded_model))

y_pred = loaded_model.predict(X_test_scaled)
print("Predictions:", y_pred)