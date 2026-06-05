import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and prepare data
data = pd.read_csv("weather-data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

X = data[['Temp_Max', 'Temp_Min', 'Humidity', 'Heatwave']]
y = data['RainFlag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Random Forest - Accuracy:", acc)

joblib.dump(model, "random_forest_model.pkl")

# Rain_mm over Date (wave)
plt.figure(figsize=(12, 5))
plt.plot(data['Date'], data['Rain_mm'], color='darkorange')
plt.fill_between(data['Date'], data['Rain_mm'], color='gold', alpha=0.4)
plt.title("🌧️ Rainfall Over Time (Random Forest)")
plt.xlabel("Date")
plt.ylabel("Rain_mm")
plt.grid(True)
plt.tight_layout()
plt.show()
