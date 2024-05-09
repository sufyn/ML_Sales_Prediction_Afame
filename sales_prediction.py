import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Load data (replace with your data path)
data = pd.read_csv("Sales.csv")

# Feature engineering (consider creating additional features based on your data)
data["Total_Advertising"] = data["TV"] + data["Radio"] + data["Newspaper"]
data["TV_Ratio"] = data["TV"] / data["Total_Advertising"]
data["Radio_Ratio"] = data["Radio"] / data["Total_Advertising"]
data["Newspaper_Ratio"] = data["Newspaper"] / data["Total_Advertising"]

# Target variable
target = "Sales"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regression": SVR()
}

# Function to evaluate and print model performance
def evaluate_model(model_name, model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n** Model: {model_name} **")
    print(f"Mean Squared Error: {mse:.2f}")

# Evaluate baseline model (without advertising features)
baseline_model = LinearRegression()
baseline_model.fit(X_train.drop(["TV", "Radio", "Newspaper"], axis=1), y_train)
y_pred_baseline = baseline_model.predict(X_test.drop(["TV", "Radio", "Newspaper"], axis=1))
baseline_mse = mean_squared_error(y_test, y_pred_baseline)
print(f"\n** Baseline Model (without Advertising Features) **")
print(f"Mean Squared Error: {baseline_mse:.2f}")

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    evaluate_model(model_name, model, X_test_scaled, y_test)

# Model optimization (example using Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Print best parameters and score
best_random_forest = grid_search.best_estimator_
print("\n** Optimized Random Forest Model **")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Mean Squared Error: {-grid_search.best_score_:.2f}")

# Evaluate the best model
evaluate_model("Optimized Random Forest", best_random_forest, X_test_scaled, y_test)

# Feature importance analysis (for Random Forest)
if isinstance(best_random_forest, RandomForestRegressor):
    feature_importance = best_random_forest.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    print("\n** Feature Importance (Optimized Random Forest) **")
    print(feature_importance_df)

# prediction
new_data = pd.DataFrame({"TV": [100], "Radio": [100], "Newspaper": [10], "Total_Advertising": [210], "TV_Ratio": [0.476], "Radio_Ratio": [0.476], "Newspaper_Ratio": [0.048]})
new_data_scaled = scaler.transform(new_data)
predicted_sales = best_random_forest.predict(new_data_scaled)
print(f"\nPredicted Sales: ${predicted_sales[0]:.2f}")

#visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance (Optimized Random Forest)")
plt.show()

# Plot actual vs. predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_random_forest.predict(X_test_scaled))
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.show()





# Save the best model (optional)
import joblib
joblib.dump(best_random_forest, "best_model.pkl")