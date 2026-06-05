import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load and prepare data
data = pd.read_csv("weather-data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

X = data[['Temp_Max', 'Temp_Min', 'Humidity', 'Heatwave']]
y = data['RainFlag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Linear Regression - Mean Squared Error:", mse)

joblib.dump(model, "linear_model.pkl")

# Rain_mm over Date (wave)
plt.figure(figsize=(12, 5))
plt.plot(data['Date'], data['Rain_mm'], color='royalblue')
plt.fill_between(data['Date'], data['Rain_mm'], color='skyblue', alpha=0.4)
plt.title("🌧️ Rainfall Over Time (Linear Regression)")
plt.xlabel("Date")
plt.ylabel("Rain_mm")
plt.grid(True)
plt.tight_layout()
plt.show()
