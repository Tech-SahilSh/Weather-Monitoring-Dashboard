import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load the data
data = pd.read_csv("weather-data.csv")

# Step 2: Select Features (X) and Target (y)
X = data[['Temp_Max', 'Temp_Min', 'Humidity', 'Heatwave']]
y = data['RainFlag']

# Step 3: Split into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Optional: Convert prediction to 0 or 1
y_pred_binary = [1 if val >= 0.5 else 0 for val in y_pred]

# Show sample prediction
print("Actual  :", list(y_test[:10]))
print("Predicted (0/1):", y_pred_binary[:10])

# Step 6: Save the model to a file
joblib.dump(model, "rain_prediction_model.pkl")
print("Model saved as rain_prediction_model.pkl")

# Step 7: Visualization
plt.scatter(data['Humidity'], data['RainFlag'], color='blue')
plt.xlabel("Humidity")
plt.ylabel("Rain (0 = No, 1 = Yes)")
plt.title("Humidity vs Rain")
plt.grid(True)
plt.show()
