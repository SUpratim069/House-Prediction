# Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the Boston Housing Dataset
try:
    boston = fetch_openml(name='boston', version=1, as_frame=True)
    data = boston.frame
    data['medv'] = boston.target  # Add target as 'medv' column for consistency
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display First Few Rows
print("Dataset Overview:")
print(data.head())

# Exploratory Data Analysis
print("\nDataset Description:")
print(data.describe())

# Check for Null Values
print("\nNull Values in Dataset:")
print(data.isnull().sum())

# Correlation Matrix and Heatmap
corr_matrix = data.corr()
corr_sorted = corr_matrix['medv'].abs().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix.loc[corr_sorted.index, corr_sorted.index], annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Sorted by Correlation with medv)')
plt.show()

# Selecting Features and Target Variable
X = data.drop(columns='medv')  # 'medv' is the target column
y = data['medv']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initializing and Training the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Abs_Coefficient', ascending=False)
print("\nModel Coefficients:")
print(coefficients[['Feature', 'Coefficient']])

# Making Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.show()

# Residuals vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Prices')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()

# Optional: QQ Plot for Residuals
import statsmodels.api as sm
sm.qqplot(residuals, line='45', fit=True)
plt.title('QQ Plot of Residuals')
plt.show()
