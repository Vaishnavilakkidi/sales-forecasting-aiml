# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Step 2: Load dataset
df = pd.read_csv("sales.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Step 3: Feature Engineering
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

X = df[["Day", "Month", "Year"]]
y = df["Sales"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Step 5: Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Step 6: Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Step 7: Model Evaluation
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name} Results:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}\n")

evaluate_model("Linear Regression", y_test, lr_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)

# Step 8: Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Sales", marker="o")
plt.plot(lr_predictions, label="Linear Regression", marker="o")
plt.plot(rf_predictions, label="Random Forest", marker="o")
plt.xlabel("Test Data Points")
plt.ylabel("Sales")
plt.title("Sales Forecasting using AI/ML")
plt.legend()
plt.grid()
plt.show()
