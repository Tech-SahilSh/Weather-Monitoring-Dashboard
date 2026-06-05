import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("weather-data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

# Linear Regression Visualization
plt.figure(figsize=(10, 4))
plt.plot(data['Date'], data['Rain_mm'], color='royalblue')
plt.fill_between(data['Date'], data['Rain_mm'], color='skyblue', alpha=0.4)
plt.title("🌧️ Rainfall Over Time (Linear Regression)")
plt.xlabel("Date")
plt.ylabel("Rain_mm")
plt.grid(True)
plt.tight_layout()
plt.savefig("linear_regression_viz.png")
plt.close()

# Logistic Regression Visualization
plt.figure(figsize=(10, 4))
plt.plot(data['Date'], data['Rain_mm'], color='seagreen')
plt.fill_between(data['Date'], data['Rain_mm'], color='lightgreen', alpha=0.4)
plt.title("🌧️ Rainfall Over Time (Logistic Regression)")
plt.xlabel("Date")
plt.ylabel("Rain_mm")
plt.grid(True)
plt.tight_layout()
plt.savefig("logistic_regression_viz.png")
plt.close()

# Random Forest Visualization
plt.figure(figsize=(10, 4))
plt.plot(data['Date'], data['Rain_mm'], color='darkorange')
plt.fill_between(data['Date'], data['Rain_mm'], color='gold', alpha=0.4)
plt.title("🌧️ Rainfall Over Time (Random Forest)")
plt.xlabel("Date")
plt.ylabel("Rain_mm")
plt.grid(True)
plt.tight_layout()
plt.savefig("random_forest_viz.png")
plt.close()

# Decision Tree Visualization
plt.figure(figsize=(10, 4))
plt.plot(data['Date'], data['Rain_mm'], color='purple')
plt.fill_between(data['Date'], data['Rain_mm'], color='violet', alpha=0.4)
plt.title("🌧️ Rainfall Over Time (Decision Tree)")
plt.xlabel("Date")
plt.ylabel("Rain_mm")
plt.grid(True)
plt.tight_layout()
plt.savefig("decision_tree_viz.png")
plt.close()
